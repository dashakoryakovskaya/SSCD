# coding: utf-8
import os, logging
import random
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.schedulers import SmartScheduler
from utils.logger_setup import color_metric, color_split
from utils.measures import mf1, uar, acc_func, ccc
from utils.losses import MultiTaskLossWithNaN_v2
from models.models import MultiModalFusionModelWithAblation
from data_loading.dataset_multimodal import MultimodalDataset


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transform_matrix(matrix):
    threshold1 = 1 - 1 / 7
    threshold2 = 1 / 7
    mask1 = matrix[:, 0] >= threshold1
    result = np.zeros_like(matrix[:, 1:])
    transformed = (matrix[:, 1:] >= threshold2).astype(int)
    result[~mask1] = transformed[~mask1]
    return result


def process_predictions(pred_emo, true_emo):
    pred_emo = torch.nn.functional.softmax(pred_emo, dim=1).cpu().detach().numpy()
    pred_emo = transform_matrix(pred_emo).tolist()
    true_emo = true_emo.cpu().detach().numpy()
    true_emo = np.where(true_emo > 0, 1, 0)[:, 1:].tolist()
    return pred_emo, true_emo

@torch.no_grad()
def evaluate_epoch(model: torch.nn.Module,
                   loader: DataLoader,
                   device: torch.device,
                   cfg) -> Dict[str, float]:
    """Collect metrics over the entire loader."""
    model.eval()
    emo_preds, emo_tgts = [], []
    pkl_preds, pkl_tgts = [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        batch = drop_domains_in_batch(batch, cfg)
        out = model(batch)

        # Emotion
        logits_e = out["emotion_logits"]
        if logits_e is not None:
            y_e = batch["labels"]["emotion"]
            valid_e = ~torch.isnan(y_e).all(dim=1)
            if valid_e.any():
                p, t = process_predictions(logits_e[valid_e], y_e[valid_e])
                emo_preds.extend(p)
                emo_tgts.extend(t)

        # Personality
        preds_p = out["personality_scores"]
        if preds_p is not None:
            preds_p = preds_p.cpu()
            y_p = batch["labels"]["personality"]
            valid_p = ~torch.isnan(y_p).all(dim=1)
            if valid_p.any():
                pkl_preds.append(preds_p[valid_p].numpy())
                pkl_tgts.append(y_p[valid_p].numpy())

    metrics: dict[str, float] = {}
    if emo_tgts:
        tgt, prd = np.asarray(emo_tgts), np.asarray(emo_preds)
        metrics["mF1"] = mf1(tgt, prd)
        metrics["mUAR"] = uar(tgt, prd)
    if pkl_tgts:
        tgt, prd = np.vstack(pkl_tgts), np.vstack(pkl_preds)
        metrics["ACC"] = acc_func(tgt, prd)
        metrics["CCC"] = ccc(tgt, prd)
    return metrics

def log_and_aggregate_split(name: str,
                            loaders: dict[str, DataLoader],
                            model: torch.nn.Module,
                            device: torch.device,
                            config) -> dict[str, float]:
    logging.info(f"—— {name} metrics ——")
    all_metrics: dict[str, float] = {}

    for ds_name, loader in loaders.items():
        m = evaluate_epoch(model, loader, device, config)
        all_metrics.update({f"{k}_{ds_name}": v for k, v in m.items()})
        msg = " · ".join(color_metric(k, v) for k, v in m.items())
        logging.info(f"[{color_split(name)}:{ds_name}] {msg}")

    mf1s = [v for k, v in all_metrics.items() if k.startswith("mF1_")]
    uars = [v for k, v in all_metrics.items() if k.startswith("mUAR_")]
    accs = [v for k, v in all_metrics.items() if k.startswith("ACC_")]
    cccs = [v for k, v in all_metrics.items() if k.startswith("CCC_")]

    if mf1s and uars:
        all_metrics["mean_emo"] = float(np.mean(mf1s + uars))
    if accs and cccs:
        all_metrics["mean_pkl"] = float(np.mean(accs + cccs))

    if "mean_emo" in all_metrics or "mean_pkl" in all_metrics:
        summary_parts = []
        if "mean_emo" in all_metrics:
            summary_parts.append(color_metric("mean_emo", all_metrics["mean_emo"]))
        if "mean_pkl" in all_metrics:
            summary_parts.append(color_metric("mean_pkl", all_metrics["mean_pkl"]))
        logging.info(f"{name} Summary | " + " ".join(summary_parts))

    return all_metrics


def drop_domains_in_batch(batch: dict, config):
    """Zero out cross-domain logits according to config flags."""
    if config.single_task:
        if getattr(config, "drop_personality_domain", False) and "personality_scores" in batch:
            for mod in batch["personality_scores"]:
                batch["personality_scores"][mod] = None
        if getattr(config, "drop_emotion_domain", False) and "emotion_logits" in batch:
            for mod in batch["emotion_logits"]:
                batch["emotion_logits"][mod] = None
    return batch

def stack_core_feats(feat_dict: dict, modal: str) -> torch.Tensor:
    parts = [feat_dict[k] for k in ["last_emo_encoder_features", "last_per_encoder_features"] if k in feat_dict]
    return torch.cat(parts)

def custom_collate_fn(batch):
    """Собирает список образцов в единый батч, отбрасывая None (невалидные)."""
    filtered_batch = []
    for sample in batch:
        if sample is None or "features" not in sample:
            continue
        modalities = sample["features"].keys()
        has_all_modalities = all(sample["features"].get(m) is not None for m in modalities)
        if has_all_modalities:
            filtered_batch.append(sample)

    if not filtered_batch:
        return None

    # --------- собираем features ---------
    features = {}          # modality → Tensor([B, D])
    metas    = {}          # modality → dict списков «побочных» полей (логиты)

    modalities = filtered_batch[0]["features"].keys()

    emo_pred = {}
    per_pred = {}

    for m in modalities:
        core_vecs = []
        emo_logits = []
        per_logits = []
        for sample in filtered_batch:
            core_vecs.append(stack_core_feats(sample["features"][m], m))
            emo_logits.append(sample["features"][m]["emotion_logits"])
            per_logits.append(sample["features"][m]["personality_scores"])

        features[m] = torch.stack(core_vecs)
        emo_pred[m] = torch.stack(emo_logits)
        per_pred[m] = torch.stack(per_logits)

    # --------- labels ---------
    emo = [b["labels"]["emotion"] for b in filtered_batch]
    person = [b["labels"]["personality"] for b in filtered_batch]
    emo = torch.stack(emo)
    person = torch.stack(person)

    return {
        "features": features,
        "labels": {
            "emotion": emo,
            "personality": person,
        },
        "emotion_logits": emo_pred,
        "personality_scores": per_pred,
    }

def make_dataset_and_loader(
    config,
    split: str,
    modality_extractors,
    *,
    only_dataset: str | None = None,
):
    """
    Универсальная функция: объединяет датасеты или возвращает один при only_dataset.
    При объединении train-датасетов — использует WeightedRandomSampler для балансировки.
    """
    if not getattr(config, "datasets", None):
        raise ValueError("⛔ В конфиге не указана секция [datasets].")

    datasets = []

    for dataset_name, dataset_cfg in config.datasets.items():
        if only_dataset and dataset_name != only_dataset:
            continue
            
        csv_path = dataset_cfg["csv_path"].format(base_dir=dataset_cfg["base_dir"], split=split)
        video_dir  = dataset_cfg["video_dir"].format(base_dir=dataset_cfg["base_dir"], split=split)

        dataset = MultimodalDataset(
            csv_path=csv_path,
            video_dir=video_dir,
            config=config,
            split=split,
            modality_feature_extractors=modality_extractors,
            dataset_name=dataset_name,
            device=config.device,
        )
        datasets.append(dataset)

    if not datasets:
        raise ValueError(f"⚠️ Для split='{split}' не найдено ни одного датасета.")

    full_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    loader = DataLoader(
        full_dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn,
    )

    return full_dataset, loader

def train_multimodal(config, train_loaders, dev_loaders,test_loaders, metrics_csv_path, model_stage):
    seed_everything(config.random_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    weight_emotion        = config.weight_emotion
    weight_pers           = config.weight_pers
    pers_loss_type        = config.pers_loss_type
    emotion_loss_type     = config.emotion_loss_type
    flag_emo_weight       = config.flag_emo_weight
    weight_decay          = config.weight_decay
    momentum              = config.momentum
    lr                    = config.lr
    num_epochs            = config.num_epochs
    max_patience          = config.max_patience
    scheduler_type        = config.scheduler_type
    
    if config.single_task:
        #   0: Emo+PKL → Emo     1: Emo → Emo
        #   2: Emo+PKL → PKL     3: PKL → PKL
        slice_map = [("both", "emo"), ("emo", "emo"),
                     ("both", "pkl"), ("pkl", "pkl")]
        try:
            feature_slice, task_target = slice_map[config.single_task_id]
        except IndexError:
            raise ValueError("single_task_id must be 0-3")

        config.drop_personality_domain = feature_slice == "emo"
        config.drop_emotion_domain     = feature_slice == "pkl"
    else:
        feature_slice = task_target = None
        config.drop_personality_domain = config.drop_emotion_domain = False
    # ablation     
    ablation_config = {}
    if not config.single_task:
        modality_combinations = [
            [],  # 0 use all modalities

            # Single modalities
            ["text"],       # 1
            ["video"]      # 2
        ]

        components = [
            -1,
            "disable_graph_attn",
            "disable_cross_attn",
            "disable_emo_logit_proj",
            "disable_pkl_logit_proj",
            "disable_guide_emo",
            "disable_guide_pkl",
        ]
        ablation_config = (
            {
                "disabled_modalities": modality_combinations[config.id_ablation_type_by_modality],
                components[config.id_ablation_type_by_component]: True
            }
            if components[config.id_ablation_type_by_component] != -1
            else {"disabled_modalities": modality_combinations[config.id_ablation_type_by_modality]}
        )
    model = MultiModalFusionModelWithAblation(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_transformer_heads,
            dropout=config.dropout,
            emo_out_dim=7,
            pkl_out_dim=5,
            device=device,
            ablation_config=ablation_config,
            attention=config.attention
        ).to(device)
            
    # Оптимизатор и лосс
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr,momentum = momentum
        )
    elif config.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"⛔ Неизвестный оптимизатор: {config.optimizer}")

    logging.info(f"Используем оптимизатор: {config.optimizer}, learning rate: {lr}")
            
    # LR Scheduler
    steps_per_epoch = sum(1 for b in train_loaders if b is not None)
    scheduler = SmartScheduler(
        scheduler_type=scheduler_type,
        optimizer=optimizer,
        config=config,
        steps_per_epoch=steps_per_epoch
    )   

    # Loss   
    criterion = MultiTaskLossWithNaN_v2(
        weight_emotion=weight_emotion,
        weight_personality=weight_pers, 
        emo_weights = (torch.FloatTensor(
                [5.890161, 7.534918, 11.228363, 27.722221, 1.3049748, 5.6189237, 26.639517]
            ).to(device) if flag_emo_weight else None),
        personality_loss_type=pers_loss_type,
        emotion_loss_type=config.emotion_loss_type,
        ssl_weight_emotion = config.ssl_weight_emotion,
        ssl_weight_personality = config.ssl_weight_personality,
        ssl_confidence_threshold_emo=config.ssl_confidence_threshold_emo,
        ssl_confidence_threshold_pt=config.ssl_confidence_threshold_pt
    ).to(device)
            
    best_score = float("-inf")
    best_dev_metrics = {}
    best_test_metrics = {}
    patience_counter = 0
    for epoch in range(num_epochs):
        logging.info(f"\n=== Эпоха {epoch} ===")
        model.train()
        total_loss = 0.0
        total_samples = 0
        total_preds_emo = []
        total_targets_emo = []
        total_preds_per = []
        total_targets_per = []
        for batch_idx, batch in enumerate(tqdm(train_loaders)):
            if batch is None:
                continue
            batch = drop_domains_in_batch(batch, config)
            
            emo_labels = batch['labels'].get('emotion')
            emo_labels = emo_labels.to(device) if emo_labels is not None else None
            per_labels = batch['labels'].get('personality')
            per_labels = per_labels.to(device) if per_labels is not None else None

            valid_emo = (~torch.isnan(emo_labels).all(dim=1)) if emo_labels is not None else None
            valid_per = (~torch.isnan(per_labels).all(dim=1)) if per_labels is not None else None

            outputs = model(batch)
            loss_labels = {}
            if emo_labels is not None:
                loss_labels["emotion"] = emo_labels
                loss_labels["valid_emo"] = valid_emo
            if per_labels is not None:
                loss_labels["personality"] = per_labels
                loss_labels["valid_per"] = valid_per

            total_task_loss = criterion(
                outputs,
                loss_labels
            )

            total_task_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(batch_level=True)
            
            if emo_labels is not None:
                bs = emo_labels.shape[0]
            elif per_labels is not None:
                bs = per_labels.shape[0]
            else:
                bs = next(iter(batch["features"].values())).shape[0]

            total_loss += float(total_task_loss.item()) * bs
            total_samples += bs
            
            if outputs.get('emotion_logits') is not None and valid_emo is not None and valid_emo.any():
                preds_emo, targets_emo = process_predictions(
                    outputs['emotion_logits'][valid_emo],
                    emo_labels[valid_emo]
                )
                total_preds_emo.extend(preds_emo)
                total_targets_emo.extend(targets_emo)
            if outputs.get('personality_scores') is not None and valid_per is not None and valid_per.any():
                preds_per = outputs['personality_scores'][valid_per]
                targets_per = per_labels[valid_per]
                total_preds_per.extend(preds_per.cpu().detach().numpy().tolist())
                total_targets_per.extend(targets_per.cpu().detach().numpy().tolist())
            # --- train metrics ---
        train_loss = total_loss / max(1, total_samples)

        # EMO
        if total_targets_emo:
            mF1_train = mf1(np.asarray(total_targets_emo), np.asarray(total_preds_emo))
            mUAR_train = uar(np.asarray(total_targets_emo), np.asarray(total_preds_emo))
            mean_emo_train = np.mean([mF1_train, mUAR_train])
        else:
            mF1_train = mUAR_train = mean_emo_train = float('nan')

        # PKL (personality)
        if total_targets_per:
            t_per = np.asarray(total_targets_per)
            p_per = np.asarray(total_preds_per)
            acc_train = acc_func(t_per, p_per)
            ccc_vals = []
            for i in range(t_per.shape[1]):
                mask = ~np.isnan(t_per[:, i])
                if mask.sum() == 0:
                    continue
                ccc_vals.append(ccc(t_per[mask, i], p_per[mask, i]))
            ccc_train = float(np.mean(ccc_vals)) if ccc_vals else float('nan')
            mean_pkl_train = np.nanmean([acc_train, ccc_train])
        else:
            acc_train = ccc_train = mean_pkl_train = float('nan')


        parts = [
            f"Loss={train_loss:.4f}",
            f"EMO: UAR={mUAR_train:.4f} MF1={mF1_train:.4f} MEAN={mean_emo_train:.4f}",
            f"PKL: ACC={acc_train:.4f} CCC={ccc_train:.4f} MEAN={mean_pkl_train:.4f}"
        ]
        logging.info(f"[{color_split('TRAIN')}] " + " | ".join(parts))

        # ── Evaluation ──
        cur_dev = log_and_aggregate_split("Dev", dev_loaders, model, device, config)
        cur_test = log_and_aggregate_split("Test", test_loaders, model, device, config) if test_loaders else {}

        cur_eval = cur_dev if config.early_stop_on == "dev" else cur_test

        mean_emo = cur_eval.get("mean_emo")
        mean_pkl = cur_eval.get("mean_pkl", 0.0)

        # ── choose target metric depending on mode ──
        if config.single_task:
            metric_val = cur_eval["mean_emo"] if task_target == "emo" else cur_eval["mean_pkl"]
        else:
            if mean_emo is not None and mean_pkl is not None:
                metric_val = 0.5 * (mean_emo + mean_pkl)
            else:
                metric_val = mean_emo if mean_emo is not None else mean_pkl

        scheduler.step(metric_val)

        if metric_val > best_score:
            best_dev_metrics = {
                "mean": 0.5 * (cur_dev.get("mean_emo") + cur_dev.get("mean_pkl", 0.0)),
                "by_dataset": {
                    "name": "all",
                    **cur_dev
                }
            }
            best_test_metrics = {
                "mean": 0.5 * (cur_test.get("mean_emo") + cur_test.get("mean_pkl", 0.0)),
                "by_dataset": {
                    "name": "all",
                    **cur_test
                }
            }
            best_score = metric_val
            best_dev = cur_dev
            best_test = cur_test
            patience_counter = 0

            os.makedirs(metrics_csv_path, exist_ok=True)
            emo_str = f"{mean_emo:.4f}" if mean_emo is not None else "NA"
            pkl_str = f"{mean_pkl:.4f}" if mean_pkl is not None else "NA"

            ckpt_path = Path(metrics_csv_path) / f"best_ep{epoch + 1}_emo{emo_str}_pkl{pkl_str}.pt"

            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"💾 Модель сохранена: {ckpt_path.name}")
        else:
            patience_counter += 1
            logging.warning(f"No improvement — patience {patience_counter}/{max_patience}")
            if patience_counter >= max_patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

    return metric_val, best_dev_metrics, best_test_metrics

                