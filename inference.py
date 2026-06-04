# coding: utf-8
import os, logging
import numpy as np
import torch
from utils.config_loader import ConfigLoader
import whisper
from models.models import MultiModalFusionModelWithAblation
from modalities.vlm.feature_extractor import PretrainedVLMEmbeddingExtractor
from modalities.text.feature_extractor import PretrainedTextEmbeddingExtractor

def aggregate(feats, average: bool = True):
        if feats is None:
            return None

        if isinstance(feats, torch.Tensor):
            if average and feats.ndim == 3:
                feats = feats.mean(dim=1)  # → [B, D]
            return feats.squeeze()

        if isinstance(feats, dict):
            return {
                key: aggregate(val, average)
                for key, val in feats.items()
            }

        raise TypeError(f"Unsupported feature type: {type(feats)}")

def transcribe_video(video_path):
    whisper_model = whisper.load_model("base")
    try:
        result = whisper_model.transcribe(video_path, fp16=False)
        return result.get('text', '')
    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        return ""

def stack_core_feats(feat_dict: dict, modal: str) -> torch.Tensor:
    parts = [feat_dict[k] for k in ["last_emo_encoder_features", "last_per_encoder_features"] if k in feat_dict]
    return torch.cat(parts)


def custom_collate_fn(batch):
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

    features = {} 
    metas    = {}

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

    return {
        "features": features,
        "emotion_logits": emo_pred,
        "personality_scores": per_pred,
    }

def predict(video_path):
    base_config = ConfigLoader("config_copy.toml")
    vlm_feature_extractor = PretrainedVLMEmbeddingExtractor(device=base_config.device)
    text_feature_extractor = PretrainedTextEmbeddingExtractor(device=base_config.device)
    modality_extractors = {
        "video": vlm_feature_extractor,
        "text": text_feature_extractor,
    }
    ablation_config = {}
    if not base_config.single_task:
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
                "disabled_modalities": modality_combinations[base_config.id_ablation_type_by_modality],
                components[base_config.id_ablation_type_by_component]: True
            }
            if components[base_config.id_ablation_type_by_component] != -1
            else {"disabled_modalities": modality_combinations[base_config.id_ablation_type_by_modality]}
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalFusionModelWithAblation(
            hidden_dim=base_config.hidden_dim,
            num_heads=base_config.num_transformer_heads,
            dropout=base_config.dropout,
            emo_out_dim=7,
            pkl_out_dim=5,
            device=device,
            ablation_config=ablation_config,
            attention=base_config.attention
        ).to(device)
    checkpoint = torch.load("multimodal_model.pt", map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    entry = {
        "video_path": video_path,
        "features": {},
    }
    try:
        video_feats = modality_extractors["video"].extract(video_path=video_path, saved=False)
        entry["features"]["video"] = aggregate(video_feats, True)
    except Exception as e:
        logging.warning(f"Video extract error {video_path}: {e}")
        entry["features"]["video"] = None
    try:
        txt_raw = transcribe_video(video_path)
        text_feats = modality_extractors["text"].extract(txt_raw)
        entry["features"]["text"] = aggregate(text_feats, True)
    except Exception as e:
        logging.warning(f"Text extract error {txt_raw}: {e}")
        entry["features"]["text"] = None

    outputs = model(custom_collate_fn([entry]))
    preds_emo = None
    preds_per = None
    if outputs.get('emotion_logits') is not None:
        preds_emo = list(torch.nn.functional.softmax(outputs['emotion_logits'], dim=1).cpu().detach().numpy())
    if outputs.get('personality_scores') is not None:
        preds_per = list(outputs['personality_scores'].cpu().detach().numpy())
    return preds_emo, preds_per
