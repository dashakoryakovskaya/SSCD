# coding: utf-8
import logging
import os
import shutil
import datetime
import toml
import torch
import io
import pickle

from utils.config_loader import ConfigLoader
from torch.utils.data import ConcatDataset, DataLoader
from utils.logger_setup import setup_logger
from utils.search_utils import greedy_search, exhaustive_search
from training.train_multi_modal import (
    make_dataset_and_loader,
    train_multimodal
)
from modalities.vlm.feature_extractor import PretrainedVLMEmbeddingExtractor
from modalities.text.feature_extractor import PretrainedTextEmbeddingExtractor

def main():
    #  Грузим конфиг
    base_config = ConfigLoader("results_biformer_2026-03-28_23-53-24/config_copy.toml")

    model_name = base_config.model_name.replace("/", "_").replace(" ", "_").lower()
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = f"results_{model_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    epochlog_dir = os.path.join(results_dir, "metrics_by_epoch")
    os.makedirs(epochlog_dir, exist_ok=True)

    # Настраиваем logging
    log_file = os.path.join(results_dir, "session_log.txt")
    setup_logger(logging.INFO, log_file=log_file)

    base_config.show_config()

    shutil.copy("results_biformer_2026-03-28_23-53-24/config_copy.toml", os.path.join(results_dir, "config_copy.toml"))
    shutil.copy("search_params_multimodal.toml", os.path.join(results_dir, "search_params_copy.toml"))
    overrides_file = os.path.join(results_dir, "overrides.txt")
    csv_prefix = os.path.join(epochlog_dir, "metrics_epochlog")

    logging.info("🔧 Initializing modalities...")

    vlm_feature_extractor = PretrainedVLMEmbeddingExtractor(device=base_config.device)
    logging.info("PretrainedVLMEmbeddingExtractor initialized")

    text_feature_extractor = PretrainedTextEmbeddingExtractor(device=base_config.device)
    logging.info("PretrainedTextEmbeddingExtractor initialized")

    modality_extractors = {
        "video": vlm_feature_extractor,
        "text": text_feature_extractor,
    }
    # Раздельные dev/test
    dev_loaders = {}
    test_loaders = {}
    train_loaders = {}
    train_datasets = []

    for dataset_name in base_config.datasets:
        train_ds, train_loader = make_dataset_and_loader(base_config, "train", modality_extractors, only_dataset=dataset_name)
        if os.path.exists(base_config.datasets[dataset_name]["csv_path"].format(base_dir=base_config.datasets[dataset_name]["base_dir"], split="dev")):
            _, dev_loader = make_dataset_and_loader(base_config, "dev",  modality_extractors, only_dataset=dataset_name)
        else:
            _, dev_loader = make_dataset_and_loader(base_config, "val",  modality_extractors, only_dataset=dataset_name)        
        if os.path.exists(base_config.datasets[dataset_name]["csv_path"].format(base_dir=base_config.datasets[dataset_name]["base_dir"], split="test")):
            _, test_loader = make_dataset_and_loader(base_config, "test",  modality_extractors, only_dataset=dataset_name)
        else:
            test_loader = dev_loader

        train_loaders[dataset_name] = train_loader
        train_datasets.append(train_ds)
        dev_loaders[dataset_name] = dev_loader
        test_loaders[dataset_name] = test_loader

    if base_config.prepare_only:
        logging.info("== Режим prepare_only: только подготовка данных, без обучения ==")
        return

    search_config = toml.load("search_params_multimodal.toml")
    param_grid = dict(search_config["grid"])
    default_values = dict(search_config["defaults"])
    
    union_train_ds = ConcatDataset(train_datasets)
    # Reuse collate_fn from any of the original loaders (identical across datasets)
    sample_loader = next(iter(train_loaders.values()))
    union_train_loader = DataLoader(
        union_train_ds,
        batch_size=base_config.batch_size,
        shuffle=True,
        num_workers=base_config.num_workers,
        collate_fn=sample_loader.collate_fn,
    )


    if base_config.search_type == "greedy":
        greedy_search(
            base_config       = base_config,
            train_loader      = union_train_loader,
            dev_loader        = dev_loaders,
            test_loader       = test_loaders,
            train_fn          = train_multimodal,
            overrides_file    = overrides_file,
            param_grid        = param_grid,
            default_values    = default_values,
            csv_prefix        = csv_prefix,
        )

    elif base_config.search_type == "exhaustive":
        exhaustive_search(
            base_config       = base_config,
            train_loader      = union_train_loader,
            dev_loader        = dev_loaders,
            test_loader       = test_loaders,
            train_fn          = train_multimodal,
            overrides_file    = overrides_file,
            param_grid        = param_grid,
            csv_prefix        = csv_prefix,

        )

    elif base_config.search_type == "none":
        logging.info("== Режим одиночной тренировки (без поиска параметров) ==")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file_path = f"{csv_prefix}_single_{timestamp}.csv"

        train_multimodal(
            config           = base_config,
            train_loaders     = union_train_loader,
            dev_loaders      = dev_loaders,
            test_loaders     = test_loaders,
            metrics_csv_path = csv_file_path,
            model_stage=base_config.model_stage
            
        )

    else:
        raise ValueError(f"⛔️ Неверное значение search_type в конфиге: '{base_config.search_type}'. Используй 'greedy', 'exhaustive' или 'none'.")


if __name__ == "__main__":
    main()
