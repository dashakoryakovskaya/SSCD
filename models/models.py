import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import logging
import copy

from .help_layers import TransformerEncoderLayer, CustomMambaBlock

from .attention.Model_CrossMPT import (
    MultiHeadedAttention,
    PositionwiseFeedForward,
    Encoder,
    EncoderLayer,
)


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
    def __init__(self, emo_model, per_model, input_dim_emotion=512, input_dim_personality=512, hidden_dim=128, out_features=512, mamba_layer_number=2, mamba_d_model=256, per_activation="sigmoid", positional_encoding=True, num_transformer_heads=4, tr_layer_number=1, dropout=0.1, num_emotions=7, num_traits=5, device='cpu'):
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


class ModalityProjector(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class AdapterFusion(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.layernorm(x + self.adapter(x))


class GuideBank(nn.Module):
    def __init__(self, out_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(out_dim, hidden_dim))

    def forward(self):
        return self.embeddings


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim=None, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        out_dim = out_dim or in_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, h, adj):
        """
        h:   [B, N, D]
        adj: [B, N, N] binary mask
        """
        B, N, D = h.size()
        Wh = self.W(h)                                # [B, N, D']
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D']
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D']
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)     # [B, N, N, 2D']
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [B, N, N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # mask non-neighbors
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)         # [B, N, D']
        return h_prime


class FeatureSlice(nn.Module):
    """
    Объединение векторов [emo‖pkl]:
      - mode='both' → [emo‖pkl]
      - mode='emo'  → только левая часть (Emo)
      - mode='pkl'  → только правая часть (PKL)
    """
    def __init__(self, mode: str = "both"):
        super().__init__()
        if mode not in ("both", "emo", "pkl"):
            raise ValueError("mode должна быть 'both' | 'emo' | 'pkl'")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "both":
            return x
        half = x.size(-1) // 2
        return x[..., :half] if self.mode == "emo" else x[..., half:]


class MultiModalFusionModelWithAblation(nn.Module):
    def __init__(
        self,
        hidden_dim=512,
        num_heads=8,
        dropout=0.1,
        emo_out_dim=7,
        pkl_out_dim=5,
        device='cpu',
        ablation_config=None,
        attention=None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        self.attention = attention

        # Ablation configuration
        self.ablation_config = ablation_config or {}
        self.disabled_modalities = set(self.ablation_config.get("disabled_modalities", []))
        self.disable_graph_attn = self.ablation_config.get("disable_graph_attn", False)
        self.disable_cross_attn = self.ablation_config.get("disable_cross_attn", False)
        self.disable_emo_logit_proj = self.ablation_config.get("disable_emo_logit_proj", False)
        self.disable_pkl_logit_proj = self.ablation_config.get("disable_pkl_logit_proj", False)
        self.disable_guide_bank = self.ablation_config.get("disable_guide_bank", False)

        self.modalities = {
            'video': 1024 * 2,
            'text': 128 * 2,
        }

        self.projectors = nn.ModuleDict({
            mod: nn.Sequential(
                ModalityProjector(in_dim, hidden_dim, dropout),
                AdapterFusion(hidden_dim, dropout),
            )
            for mod, in_dim in self.modalities.items()
        })

        if not self.disable_graph_attn:
            self.graph_attn = GraphAttentionLayer(hidden_dim, dropout=dropout)
            # self.graph_attn = GraphAttentionLayer_v4(hidden_dim, dropout=dropout)

        self.emo_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pkl_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        if not self.disable_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            if self.attention == 'Cross-MPT':
                # Cross-MPT alternative
                c = copy.deepcopy
                attn = MultiHeadedAttention(num_heads, hidden_dim)
                ff = PositionwiseFeedForward(hidden_dim, hidden_dim * 4, dropout)
                self.cross_emo = Encoder(EncoderLayer(hidden_dim, c(attn), c(ff), dropout), 1)
                self.cross_pkl = Encoder(EncoderLayer(hidden_dim, c(attn), c(ff), dropout), 1)

        self.emo_head = nn.Linear(hidden_dim, emo_out_dim)
        self.pkl_head = nn.Linear(hidden_dim, pkl_out_dim)

        # Optional learned fusers (kept for reference)
        # self.emo_fusion = nn.Linear(2, 1)
        # self.pkl_fusion = nn.Linear(2, 1)

        if not self.disable_guide_bank:
            self.guide_bank_emo = GuideBank(emo_out_dim, hidden_dim)
            self.guide_bank_pkl = GuideBank(pkl_out_dim, hidden_dim)

        if not self.disable_emo_logit_proj:
            self.emo_logit_proj = nn.Linear(emo_out_dim, hidden_dim)
        if not self.disable_pkl_logit_proj:
            self.per_logit_proj = nn.Linear(pkl_out_dim, hidden_dim)

    def forward(self, batch):
        x_mods = []
        valid_modalities = []

        for mod, feat in batch['features'].items():
            if feat is not None and mod in self.projectors and mod not in self.disabled_modalities:
                x_proj = self.projectors[mod](feat.to(self.device))  # [B, D]
                x_mods.append(x_proj)
                valid_modalities.append(mod)

        if not x_mods:
            raise ValueError("No valid modality features found")

        x_mods = torch.stack(x_mods, dim=1)  # [B, N, D]
        B, N, D = x_mods.size()

        if self.disable_graph_attn:
            context = x_mods
        else:
            adj = torch.ones(B, N, N, device=self.device)
            context = self.graph_attn(x_mods, adj)  # [B, N, D]

        emo_q = self.emo_query.expand(B, 1, -1)  # [B, 1, D]
        pkl_q = self.pkl_query.expand(B, 1, -1)  # [B, 1, D]

        if self.disable_cross_attn:
            emo_repr = context.mean(dim=1)
            pkl_repr = context.mean(dim=1)
        else:
            if self.attention == 'Cross-MPT':
                emb1_e, emb2_e = self.cross_emo(emo_q, context, None, None)   # [B,1,D], [B,N,D]
                emo_repr = torch.cat([emb1_e, emb2_e], dim=1).mean(dim=1)     # [B,D]
                emb1_p, emb2_p = self.cross_pkl(pkl_q, context, None, None)   # [B,1,D], [B,N,D]
                pkl_repr = torch.cat([emb1_p, emb2_p], dim=1).mean(dim=1)     # [B,D]
            else:
                emo_repr, _ = self.cross_attn(emo_q, context, context)
                pkl_repr, _ = self.cross_attn(pkl_q, context, context)
                emo_repr = emo_repr.squeeze(1)
                pkl_repr = pkl_repr.squeeze(1)         


        emo_logit_feats = []
        per_logit_feats = []
        for mod in valid_modalities:
            emo_logit_feats.append(batch['emotion_logits'][mod].to(self.device))
            per_logit_feats.append(batch['personality_scores'][mod].to(self.device))

        if emo_logit_feats and not self.disable_emo_logit_proj:
            emo_repr += self.emo_logit_proj(torch.stack(emo_logit_feats).mean(dim=0))
        if per_logit_feats and not self.disable_pkl_logit_proj:
            pkl_repr += self.per_logit_proj(torch.stack(per_logit_feats).mean(dim=0))

        emo_pred = self.emo_head(emo_repr)
        pkl_pred = torch.sigmoid(self.pkl_head(pkl_repr))

        if not self.disable_guide_bank:
            if not self.ablation_config.get("disable_guide_emo", False):
                guides_emo = self.guide_bank_emo()  # [emo_out_dim, D]
                emo_sim = F.cosine_similarity(emo_repr.unsqueeze(1), guides_emo.unsqueeze(0), dim=-1)
                # emo_stack = torch.stack([emo_pred, emo_sim], dim=-1)  # [B, C, 2]
                # emo_final = self.emo_fusion(emo_stack).squeeze(-1)    # [B, C]
                emo_final = (emo_pred + emo_sim) / 2
            else:
                emo_final = emo_pred

            if not self.ablation_config.get("disable_guide_pkl", False):
                guides_pkl = self.guide_bank_pkl()  # [pkl_out_dim, D]
                pkl_sim = F.cosine_similarity(pkl_repr.unsqueeze(1), guides_pkl.unsqueeze(0), dim=-1)
                # pkl_stack = torch.stack([pkl_pred, torch.sigmoid(pkl_sim)], dim=-1)
                # pkl_final = self.pkl_fusion(pkl_stack).squeeze(-1)
                pkl_final = (pkl_pred + torch.sigmoid(pkl_sim)) / 2
            else:
                pkl_final = pkl_pred
        else:
            emo_final = emo_pred
            pkl_final = pkl_pred

        return {'emotion_logits': emo_final, "personality_scores": pkl_final}
