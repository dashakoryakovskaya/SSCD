#!/usr/bin/env python
# coding: utf-8
from os import listdir
from os.path import isfile, join
import pickle
import torch
import gc

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    dtype="auto", 
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

model.config.output_hidden_states=True
model.config.text_config.output_hidden_states=True
path = '../data/CMU-MOSEI/video/train'
path_emb = '../data/CMU-MOSEI_Qwen3-VL-4B-Instruct/hidden_states/train'
with open('video_name','rb') as file:
    video_name = pickle.load(file)
for video_files in video_name.values():
    for video_file in video_files:
        conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": join(path, video_file)},
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
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(conversation, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
        inputs = inputs.to(model.device)
        with torch.no_grad():
            hidden_states = model(**inputs).hidden_states
        with open(join(path_emb, video_file[:-4] + '.pkl'), 'wb') as file:
            pickle.dump(hidden_states, file)
