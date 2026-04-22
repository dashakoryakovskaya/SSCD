# coding: utf-8
import torch
import os
import io
import logging
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import gc
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from .model_loader import load_fusion_model

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


class PretrainedVLMEmbeddingExtractor:

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        fusion_ckpt: str = "modalities/vlm/checkpoints_models/Transformer_vlm_fusion.pt",
        emo_ckpt: str   = "modalities/vlm/checkpoints_models/Transformer_vlm_emotion.pt",
        per_ckpt: str   = "modalities/vlm/checkpoints_models/Transformer_vlm_personality.pt",
    ):
        self.device = torch.device(device)
        
        '''self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, 
            dtype="auto", 
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.config.output_hidden_states=True
        self.model.config.text_config.output_hidden_states=True
        self.model.eval()'''

        self.fusion, _ = load_fusion_model(
            fusion_ckpt, emo_ckpt, per_ckpt, device=self.device
        )
        self.meta = {}
        for dataset_name in ['fiv2', 'cmu_mosei']:
            for split in ['train', 'test', 'dev']:
                meta_filename = '{}_{}_seed_42_subset_size_0.pickle'.format(dataset_name, split)
                pickle_path = os.path.join('./features', meta_filename)
                self.load_data(pickle_path, key_dict = meta_filename)

    def load_data(self, filename, key_dict = None):
        if key_dict is None:
            if os.path.exists(filename):
                if torch.cuda.is_available():
                    with open(filename, 'rb') as handle:
                        self.meta = pickle.load(handle)
                else:
                    with open(filename, 'rb') as handle:
                        self.meta = CPU_Unpickler(handle).load()
            else:
                self.meta = []
        else:
            if os.path.exists(filename):
                if torch.cuda.is_available():
                    with open(filename, 'rb') as handle:
                        self.meta[key_dict] = pickle.load(handle)
                else:
                    with open(filename, 'rb') as handle:
                        self.meta[key_dict] = CPU_Unpickler(handle).load()

    @torch.no_grad()
    def extract(self, video_path: str, dataset_name: str = "cmu_mosei", split: str = "train", save_feature_path: str = "./features", saved=True) -> dict:
        if not saved:
            conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "min_pixels": 4 * 32 * 32, "max_pixels": 64 * 32 * 32, "fps": 1.0},
                    {"type": "text", "text": 
            """You are an expert emotion analysis system.
            1) Describe the video based on characteristics and facts, important definitions of emotions and personal traits of a person.
            2) Analyze video and predict the most pronounced of the following emotions: neutral, happy, sad, anger, surprise, disgust, fear.
            3) For every of the following 5 personal traits (openness, conscientiousness, extraversion, agreeableness, non-neuroticism) determine its expression from 0 to 1.

            Output format:
            Description of the video
            Most pronounced emotion (or several emotions) separated by commas
            Expression of the 5 personal traits: {'openness': openness_expression, 'conscientiousness': conscientiousness_expression, 'extraversion': extraversion_expression, 'agreeableness': agreeableness_expression, 'non-neuroticism': non-neuroticism'_expression} """}
                ],
            },
            ]
            text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(conversation, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            inputs = self.processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
            inputs = inputs.to(model.device)
            with torch.no_grad():
                hidden_states = model(**inputs).hidden_states
            vlm_features = (hidden_states[36][0] + hidden_states[35][0] + hidden_states[34][0] + hidden_states[6][0]) / 4
        else:
            #dataset_dict = {'cmu_mosei': 'CMU-MOSEI', 'fiv2': 'FirstImpressionsV2'}
            #logging.info(f"video_path = {video_path}")
            
            for x in self.meta['{}_{}_seed_42_subset_size_0.pickle'.format(dataset_name, split)]:
                #logging.info(f"x['video_path'] = {x['video_path']}")
                if x['video_path'] == video_path:
                    vlm_features = x['vlm'].unsqueeze(0)
                    break
            
        out = self.fusion(
            emotion_input=vlm_features,
            personality_input=vlm_features,
            return_features=True,
        )

        return {
            "emotion_logits": out["emotion_logits"].cpu(),
            "personality_scores": out["personality_scores"].cpu(),
            "last_emo_encoder_features": out["last_emo_encoder_features"].cpu(),
            "last_per_encoder_features": out["last_per_encoder_features"].cpu(),
        }
