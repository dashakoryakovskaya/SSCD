# coding: utf-8
import os
import io
import logging
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


class MultimodalDataset(Dataset):
    """
    Датасет для формирования обучающих данных по всем модальностям.
    """
    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        config,
        split: str,
        modality_feature_extractors: dict,
        dataset_name: str,
        device: str = "cuda",
    ):
        """
        :param csv_path: Путь к CSV-файлу.
        :param video_dir: Путь к видео
        :param config: Конфиг
        :param split: "train", "dev" или "test".
        :param modality_feature_extractors
        :param subset_size: Если > 0, используется только первые N примеров из CSV (для отладки).
        :param dataset_name: Название корпуса
        """
        super().__init__()

        self.csv_path          = csv_path
        self.video_dir         = video_dir
        self.config            = config
        self.split             = split
        self.dataset_name      = dataset_name
        self.device            = device
        self.subset_size       = config.subset_size
        self.average_features  = config.average_features
        self.extractors: dict[str, object] = modality_feature_extractors


        if self.dataset_name == 'cmu_mosei':
            self.emotion_columns = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
            self.personality_columns  = []
        elif self.dataset_name == 'fiv2':
            self.personality_columns = ["openness", "conscientiousness", "extraversion", "agreeableness", "non-neuroticism"]
            self.emotion_columns = []
        else:
            raise ValueError(f"Название корпуса {self.dataset_name} не соотвествует целевому!")


        self.save_prepared_data = config.save_prepared_data
        self.save_feature_path  = config.save_feature_path
        self.feature_filename   = (
            f"{self.dataset_name}_{self.split}"
            f"_seed_{config.random_seed}_subset_size_{self.subset_size}"
            f"_average_features_{self.average_features}_feature_norm_{config.emb_normalize}.pickle"
        )
        self.num_emotion = 7
        self.num_personality = 5

        # Загружаем CSV
        if not os.path.exists(csv_path):
            raise ValueError(f"Ошибка: файл CSV не найден: {csv_path}")
        self.df = pd.read_csv(csv_path).dropna()
        self.df = self.df.rename(columns={
            "video_name": "filename",
        })
        if self.subset_size > 0:
            self.df = self.df.head(self.subset_size)
            logging.info(f"[DatasetMultiModal] Используем только {len(self.df)} записей (subset_size={self.subset_size}).")

        self.meta: list[dict] = []

        if self.save_prepared_data:
            os.makedirs(self.save_feature_path, exist_ok=True)
            self.pickle_path = os.path.join(self.save_feature_path, self.feature_filename)
            self.load_data(self.pickle_path)

            if not self.meta:
                self.prepare_data()
                self.save_data(self.pickle_path)
        else:
            self.prepare_data()

    def save_data(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self.meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, filename):
        if os.path.exists(filename):
            if torch.cuda.is_available():
                with open(filename, 'rb') as handle:
                    self.meta = pickle.load(handle)
            else:
                with open(filename, 'rb') as handle:
                    self.meta = CPU_Unpickler(handle).load()
        else:
            self.meta = []

    def find_file(self, base_dir: str, base_filename: str):
        for root, _, files in os.walk(base_dir):
            for file in files:
                if os.path.splitext(file)[0] == base_filename:
                    return os.path.join(root, file)
        return None

    def aggregate(self, feats, average: bool = None):
        """
        Усреднение эмбеддингов по временному измерению.
        """

        if average is None:
            average = self.average_features

        if feats is None:
            return None

        if isinstance(feats, torch.Tensor):
            if average and feats.ndim == 3:
                feats = feats.mean(dim=1)  # → [B, D]
            return feats.squeeze()

        if isinstance(feats, dict):
            return {
                key: self.aggregate(val, average)
                for key, val in feats.items()
            }

        raise TypeError(f"Unsupported feature type: {type(feats)}")
        
    def prepare_data(self):
        for index, row in tqdm(self.df.iterrows(), desc="Extracting multimodal features"):
            video_path = self.find_file(self.video_dir, row["filename"])
            if video_path is None:
                print(f"❌ Video not found: {name}")
                continue
            entry = {
                "sample_name": row["filename"],
                "video_path": video_path,
                "features": {},
            }

            # video
            try:
                video_feats = self.extractors["video"].extract(video_path=video_path, dataset_name=self.dataset_name, split=self.split, save_feature_path=self.save_feature_path)
                entry["features"]["video"] = self.aggregate(video_feats, self.average_features)
            except Exception as e:
                logging.warning(f"Video extract error {row["filename"]}: {e}")
                entry["features"]["video"] = None

            # text
            try:
                txt_raw = self.df[self.df["filename"] == row["filename"]]["text"].values[0]
                text_feats = self.extractors["text"].extract(txt_raw)
                entry["features"]["text"] = self.aggregate(text_feats, self.average_features)
            except Exception as e:
                logging.warning(f"Text extract error {row["filename"]}: {e}")
                entry["features"]["text"] = None

            # labels
            try:
                emotion_tensor     = None
                personality_tensor = None

                #   ─ emotion ─
                if self.emotion_columns:
                    emotion_tensor = torch.tensor(
                        self.df.loc[
                            self.df["filename"] == row["filename"], self.emotion_columns
                        ].values[0],
                        dtype=torch.float32
                    )
                else:
                    emotion_tensor = torch.full(
                        (self.num_emotion,), torch.nan, dtype=torch.float32
                    )

                #   ─ personality ─
                if self.personality_columns:
                    personality_tensor = torch.tensor(
                        self.df.loc[
                            self.df["filename"] == row["filename"], self.personality_columns
                        ].values[0],
                        dtype=torch.float32
                    )
                else:
                    personality_tensor = torch.full(
                        (self.num_personality,), torch.nan, dtype=torch.float32
                    )

                entry["labels"] = {
                    "emotion":     emotion_tensor,
                    "personality": personality_tensor
                }

            except Exception as e:
                logging.warning(f"Label extract error {row["filename"]}: {e}")
                entry["labels"] = {
                    "emotion":     torch.full((self.num_emotion,), torch.nan, dtype=torch.float32),
                    "personality": torch.full((self.num_personality,), torch.nan, dtype=torch.float32)
                }

            self.meta.append(entry)
            torch.cuda.empty_cache()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.meta[index]
