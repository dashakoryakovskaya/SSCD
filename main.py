# coding: utf-8
import logging
import os
import shutil
import datetime
import toml

from utils.config_loader import ConfigLoader
from utils.logger_setup import setup_logger
from utils.search_utils import greedy_search, exhaustive_search
from training.train_uni_modal import (
    make_dataset_and_loader,
    train_once
)

def main():
    #  Грузим конфиг
    base_config = ConfigLoader("config.toml")

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

    shutil.copy("config.toml", os.path.join(results_dir, "config_copy.toml"))
    overrides_file = os.path.join(results_dir, "overrides.txt")
    csv_prefix = os.path.join(epochlog_dir, "metrics_epochlog")

    text_feature_extractor = None
    vlm = None
    # Раздельные dev/test
    dev_loaders = {}
    test_loaders = {}
    train_loaders = {}

    for dataset_name in base_config.datasets:
        _, train_loader = make_dataset_and_loader(base_config, "train", text_feature_extractor, vlm, only_dataset=dataset_name)
        if os.path.exists(base_config.datasets[dataset_name]["csv_path"].format(base_dir=base_config.datasets[dataset_name]["base_dir"], split="dev")):
            _, dev_loader = make_dataset_and_loader(base_config, "dev",  text_feature_extractor, vlm, only_dataset=dataset_name)
        else:
            _, dev_loader = make_dataset_and_loader(base_config, "val",  text_feature_extractor, vlm, only_dataset=dataset_name)        
        if os.path.exists(base_config.datasets[dataset_name]["csv_path"].format(base_dir=base_config.datasets[dataset_name]["base_dir"], split="test")):
            _, test_loader = make_dataset_and_loader(base_config, "test",  text_feature_extractor, vlm, only_dataset=dataset_name)
        else:
            test_loader = dev_loader

        train_loaders[dataset_name] = train_loader
        dev_loaders[dataset_name] = dev_loader
        test_loaders[dataset_name] = test_loader

    if base_config.prepare_only:
        logging.info("== Режим prepare_only: только подготовка данных, без обучения ==")
        return

    search_config = toml.load("search_params.toml")
    param_grid = dict(search_config["grid"])
    default_values = dict(search_config["defaults"])

    if base_config.search_type == "greedy":
        greedy_search(
            base_config       = base_config,
            train_loader      = train_loaders,
            dev_loader        = dev_loaders,
            test_loader       = test_loaders,
            train_fn          = train_once,
            overrides_file    = overrides_file,
            param_grid        = param_grid,
            default_values    = default_values,
            csv_prefix        = csv_prefix,
            model_stage       = base_config.model_stage
        )

    elif base_config.search_type == "exhaustive":
        exhaustive_search(
            base_config       = base_config,
            train_loader      = train_loaders,
            dev_loader        = dev_loaders,
            test_loader       = test_loaders,
            train_fn          = train_once,
            overrides_file    = overrides_file,
            param_grid        = param_grid,
            csv_prefix        = csv_prefix,
            model_stage       = base_config.model_stage

        )

    elif base_config.search_type == "none":
        logging.info("== Режим одиночной тренировки (без поиска параметров) ==")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file_path = f"{csv_prefix}_single_{timestamp}.csv"

        train_once(
            config           = base_config,
            train_loader     = train_loaders,
            dev_loaders      = dev_loaders,
            test_loaders     = test_loaders,
            metrics_csv_path = csv_file_path,
            model_stage      = base_config.model_stage
        )

    else:
        raise ValueError(f"⛔️ Неверное значение search_type в конфиге: '{base_config.search_type}'. Используй 'greedy', 'exhaustive' или 'none'.")


if __name__ == "__main__":
    main()
