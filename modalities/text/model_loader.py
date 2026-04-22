# coding: utf-8
import torch
from .architectures import (
    EmotionMamba,
    PersonalityMamba,
    FusionTransformer,
)


def load_pretrained_emotion_encoder(checkpoint_path, device):
    emotion_model = EmotionMamba(
        input_dim_emotion=1024,
        input_dim_personality=1024,
        hidden_dim=256,
        out_features=128,
        mamba_layer_number=2,
        dropout=0.1
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    emotion_model.load_state_dict(state_dict)

    def extract_features(inputs, lengths):
        features = emotion_model.emo_proj(inputs)
        for block in emotion_model.emotion_encoder:
            features = block(features)
        return features

    emotion_model.extract_features = extract_features
    emotion_model.eval()
    return emotion_model

def load_pretrained_personality_encoder(checkpoint_path, device):
    personality_model = PersonalityMamba(
        input_dim_emotion=1024, 
        input_dim_personality=1024, 
        hidden_dim=64, 
        out_features=256, 
        mamba_layer_number=3, 
        dropout=0.1).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    personality_model.load_state_dict(checkpoint)

    def extract_features(inputs, lengths):
        features = personality_model.per_proj(inputs)
        for block in personality_model.personality_encoder:
            features = block(features, features, features)
        return features

    personality_model.extract_features = extract_features
    personality_model.eval()
    return personality_model

def load_fusion_model(
    fusion_checkpoint_path: str,
    emotion_encoder_checkpoint: str,
    personality_encoder_checkpoint: str,
    device: str = "cpu",
):
    device = torch.device(device)

    emotion_encoder = load_pretrained_emotion_encoder(emotion_encoder_checkpoint, device)
    personality_encoder = load_pretrained_personality_encoder(personality_encoder_checkpoint, device)

    checkpoint = torch.load(fusion_checkpoint_path, map_location=device)

    fusion_model = FusionTransformer(
        emo_model=emotion_encoder,
        per_model=personality_encoder,
        hidden_dim=128,
        out_features=64,
        tr_layer_number=3,
        num_transformer_heads=16,
        dropout=0.1
    ).to(device)
    fusion_model.load_state_dict(checkpoint)
    fusion_model.eval()
    return fusion_model, device
