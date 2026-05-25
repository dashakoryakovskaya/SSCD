# coding: utf-8
import os
import logging
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import gc

class DatasetVLM(Dataset):
    """
    Датасет для формирования обучающих данных по видео.
    """

    def __init__(
        self,
        csv_path,
        video_dir,
        config,
        split,
        vlm,
        dataset_name
    ):
        """
        :param csv_path: Путь к CSV-файлу.
        :param video_dir: Путь к видео
        :param split: "train", "dev" или "test".
        :param vlm: VLM
        :param subset_size: Если > 0, используется только первые N примеров из CSV (для отладки).
        :param dataset_name: Название корпуса
        """
        super().__init__()
        self.split = split
        self.video_dir = video_dir
        self.vlm = vlm
        self.subset_size    = config.subset_size
        self.seed = config.random_seed
        self.dataset_name = dataset_name
        self.save_prepared_data = config.save_prepared_data
        self.save_feature_path = config.save_feature_path

        if self.dataset_name == 'cmu_mosei':
            self.label_columns = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
        elif self.dataset_name == 'fiv2':
            self.label_columns = ["openness", "conscientiousness", "extraversion", "agreeableness", "non-neuroticism"]
        else:
            raise ValueError(f"Название корпуса {self.dataset_name} не соотвествует целевому!")

        # Загружаем CSV
        if not os.path.exists(csv_path):
            raise ValueError(f"Ошибка: файл CSV не найден: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna()
        self.df = self.df.rename(columns={
            "video_name": "filename",
        })
        if self.subset_size > 0:
            self.need_segment_name = list(set(self.df.filename.tolist()))[:self.subset_size]
            self.df = self.df[self.df.filename.isin(self.need_segment_name)]
            logging.info(f"[DatasetVLM] Используем только {len(self.df)} записей (subset_size={self.subset_size}).")
        else:
            self.need_segment_name = list(set(self.df.filename.tolist()))

        if not os.path.exists(self.video_dir):
            raise ValueError(f"Ошибка: директория с видео {self.video_dir} не существует!")

        if self.save_prepared_data:
            self.meta = []
            meta_filename = '{}_{}_seed_{}_subset_size_{}.pickle'.format(
                self.dataset_name,
                self.split,
                self.seed,
                self.subset_size,
            )

            pickle_path = os.path.join(self.save_feature_path, meta_filename)
            self.load_data(pickle_path)

            if not self.meta:
                self.prepare_data()
                os.makedirs(self.save_feature_path, exist_ok=True)
                self.save_data(pickle_path)

    def save_data(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self.meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as handle:
                self.meta = pickle.load(handle)
        else:
            self.meta = []

    def __len__(self):
        if self.save_prepared_data:
            return len(self.meta)
        else:
            return len(self.need_segment_name)

    def find_file_recursive(self, base_dir, filename):
        for root, dirs, files in os.walk(base_dir):
            if filename in files:
                return os.path.join(root, filename)
        return None

    def get_data(self, segment_name):
        curr_data = self.df[self.df.filename==segment_name] # отбираем все строки нужного сегмента видео
        curr_data = curr_data.dropna()

        label_vec = curr_data[self.label_columns].values[0]

        full_path_video = self.find_file_recursive(self.video_dir, curr_data.filename.unique()[0])

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

        conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": full_path_video},
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
        vlm_features = (hidden_states[36][0] + hidden_states[35][0] + hidden_states[34][0] + hidden_states[6][0]) / 4
        del text
        del images
        del videos
        del hidden_states
        del inputs
        torch.cuda.reset_peak_memory_stats(device=None)
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "video_path": full_path_video,
            "vlm": vlm_features,
            "label": torch.tensor(label_vec, dtype=torch.float32),
        }

    def prepare_data(self):
        """
        Загружает и обрабатывает один элемент датасета (on-the-fly).
        """
        for idx, segment_name in enumerate(tqdm(self.need_segment_name)):
            curr_dict = self.get_data(segment_name)

            self.meta.append(
            curr_dict
            )

    def __getitem__(self, index):
        if self.save_prepared_data:
            return self.meta[index]
        else:
            return self.get_data(self.need_segment_name[index])
