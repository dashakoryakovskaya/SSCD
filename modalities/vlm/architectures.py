# coding: utf-8
import math
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .help_layers import TransformerEncoderLayer, CustomMambaBlock

class EmotionMamba(nn.Module):
    def __init__(self, input_dim_emotion=1024, input_dim_personality=1024, hidden_dim=128, out_features=512, mamba_layer_number=2, mamba_d_model=256, per_activation="sigmoid", positional_encoding=True, num_transformer_heads=4, transformer_dropout=0.1, tr_layer_number=1, dropout=0.1, num_emotions=7, num_traits=5, device='cpu'):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_dim_emotion = input_dim_emotion

        self.emo_proj = nn.Sequential(
            nn.Linear(input_dim_emotion, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.emotion_encoder = nn.ModuleList([
            CustomMambaBlock(hidden_dim, mamba_d_model, dropout=dropout)
            for _ in range(mamba_layer_number)
        ])

        self.emotion_fc_out = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_emotions)
        )

    def forward(self, emotion_input=None, personality_input=None, return_features=False):
        print(emotion_input.float().shape)
        print(self.input_dim_emotion, self.hidden_dim)
        emo = self.emo_proj(emotion_input.float())  # (B, T, hidden_dim)
        for layer in self.emotion_encoder:
            emo = layer(emo)
        out_emo = self.emotion_fc_out(emo.mean(dim=1))  # (B, num_emotions)
        if return_features:
            return {
                'emotion_logits': out_emo,
                'last_encoder_features': emo,
            }
        else:
            return {'emotion_logits': out_emo}


class EmotionTransformer(nn.Module):
    def __init__(self, input_dim_emotion=512, input_dim_personality=512, hidden_dim=128, out_features=512, mamba_layer_number=2, mamba_d_model=256, per_activation="sigmoid", positional_encoding=True, num_transformer_heads=4, tr_layer_number=1, dropout=0.1, num_emotions=7, num_traits=5, device='cpu'):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.emo_proj = nn.Sequential(
            nn.Linear(input_dim_emotion, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.emotion_encoder = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self.emotion_fc_out = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_emotions)
        )

    def forward(self, emotion_input=None, personality_input=None, return_features=False):
        emo = self.emo_proj(emotion_input.float())

        for layer in self.emotion_encoder:
            emo += layer(emo, emo, emo)

        out_emo = self.emotion_fc_out(emo.mean(dim=1))
        if return_features:
            return {
                'emotion_logits': out_emo,
                'last_encoder_features': emo,
            }
        else:
            return {'emotion_logits': out_emo}


class PersonalityTransformer(nn.Module):
    def __init__(self, input_dim_emotion=512, input_dim_personality=512, hidden_dim=128, out_features=512, mamba_layer_number=2, mamba_d_model=256, per_activation="sigmoid", positional_encoding=True, num_transformer_heads=4, tr_layer_number=1, dropout=0.1, num_emotions=7, num_traits=5, device='cpu'):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.per_proj = nn.Sequential(
            nn.Linear(input_dim_personality, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.personality_encoder = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self.personality_fc_out = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_traits)
        )

        if per_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif per_activation == "relu":
            self.activation = nn.ReLU()

    def forward(self, emotion_input=None, personality_input=None, return_features=False, activation=True):
        per = self.per_proj(personality_input.float())

        for layer in self.personality_encoder:
            per += layer(per, per, per)

        out_per = self.personality_fc_out(per.mean(dim=1))

        if return_features:
            return {
                'personality_scores': self.activation(out_per) if activation else out_per,
                'last_encoder_features': per,
            }
        else:
            return {'personality_scores': self.activation(out_per) if activation else out_per}


class PersonalityMamba(nn.Module):
    def __init__(self, input_dim_emotion=512, input_dim_personality=512, hidden_dim=128, out_features=512, mamba_layer_number=2, mamba_d_model=256, per_activation="sigmoid", positional_encoding=True, num_transformer_heads=4, tr_layer_number=1, dropout=0.1, num_emotions=7, num_traits=5, device='cpu'):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.per_proj = nn.Sequential(
            nn.Linear(input_dim_personality, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.personality_encoder = nn.ModuleList([
            CustomMambaBlock(hidden_dim, mamba_d_model, dropout=dropout)
            for _ in range(mamba_layer_number)
        ])

        self.personality_fc_out = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_traits)
        )

        if per_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif per_activation == "relu":
            self.activation = nn.ReLU()

    def forward(self, emotion_input=None, personality_input=None, return_features=False, activation=True):
        per = self.per_proj(personality_input.float())

        for layer in self.personality_encoder:
            per = layer(per)

        out_per = self.personality_fc_out(per.mean(dim=1))

        if return_features:
            return {
                'personality_scores': self.activation(out_per) if activation else out_per,
                'last_encoder_features': per,
            }
        else:
            return {'personality_scores': self.activation(out_per) if activation else out_per}


class FusionTransformer(nn.Module):
    def __init__(self, emo_model, per_model, hidden_dim=128, out_features=512, mamba_layer_number=2, mamba_d_model=256, per_activation="sigmoid", positional_encoding=True, num_transformer_heads=4, tr_layer_number=1, dropout=0.1, num_emotions=7, num_traits=5, device='cpu'):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.hidden_dim = hidden_dim

        self.emo_model = emo_model
        self.per_model = per_model

        for param in self.emo_model.parameters():
            param.requires_grad = False

        for param in self.per_model.parameters():
            param.requires_grad = False

        self.emo_proj = nn.Sequential(
            nn.Linear(self.emo_model.hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.per_proj = nn.Sequential(
            nn.Linear(self.per_model.hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.emotion_to_personality_attn = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self.personality_to_emotion_attn = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self.emotion_personality_fc_out = nn.Sequential(
            nn.Linear(hidden_dim*2, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_emotions)
        )

        self.personality_emotion_fc_out = nn.Sequential(
            nn.Linear(hidden_dim*2, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_traits)
        )

        if per_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif per_activation == "relu":
            self.activation = nn.ReLU()

    def forward(self, emotion_input=None, personality_input=None, return_features=False):
        emo_features = self.emo_model(emotion_input=emotion_input.float(), return_features=True)
        per_features = self.per_model(personality_input=personality_input.float(), return_features=True)

        emo_emd = self.emo_proj(emo_features['last_encoder_features'])
        per_emd = self.per_proj(per_features['last_encoder_features'])

        # padding
        max_len = max(emo_emd.shape[1], per_emd.shape[1])
        emo_emd = emo_emd.cpu().detach().numpy()
        per_emd = per_emd.cpu().detach().numpy()
        emo_emd = np.pad(emo_emd[:, :max_len, :], ((0, 0), (0, max(0, max_len - emo_emd.shape[1])), (0, 0)), "constant")
        per_emd = np.pad(per_emd[:, :max_len, :], ((0, 0), (0, max(0, max_len - per_emd.shape[1])), (0, 0)), "constant")
        emo_emd = torch.tensor(emo_emd, device=self.device)
        per_emd = torch.tensor(per_emd, device=self.device)

        for layer in self.emotion_to_personality_attn:
            emo_emd += layer(emo_emd, per_emd, per_emd)

        for layer in self.personality_to_emotion_attn:
            per_emd += layer(per_emd, emo_emd, emo_emd)

        fused = torch.cat([emo_emd, per_emd], dim=-1)
        emotion_logits = self.emotion_personality_fc_out(fused.mean(dim=1))
        personality_scores = self.personality_emotion_fc_out(fused.mean(dim=1))

        if return_features:
            return {
                'emotion_logits': (emotion_logits+emo_features['emotion_logits'])/2,
                'personality_scores': (self.activation(personality_scores)+per_features['personality_scores'])/2,
                'last_emo_encoder_features': emo_emd,
                'last_per_encoder_features': per_emd,
            }
        else:
            return {'emotion_logits': (emotion_logits+emo_features['emotion_logits'])/2,
                    'personality_scores': (self.activation(personality_scores)+per_features['personality_scores'])/2,}
