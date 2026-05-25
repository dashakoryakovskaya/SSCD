# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.layer_2(x)


class AddAndNorm(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        return self.norm(x + self.dropout(residual))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1)].detach()  # Отключаем градиенты
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1, positional_encoding=False):
        super().__init__()
        self.input_dim = input_dim
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = PositionWiseFeedForward(input_dim, input_dim, dropout=dropout)
        self.add_norm_after_attention = AddAndNorm(input_dim, dropout=dropout)
        self.add_norm_after_ff = AddAndNorm(input_dim, dropout=dropout)
        self.positional_encoding = PositionalEncoding(input_dim) if positional_encoding else None

    def forward(self, query, key, value):
        if self.positional_encoding:
            key = self.positional_encoding(key)
            value = self.positional_encoding(value)
            query = self.positional_encoding(query)

        attn_output, _ = self.self_attention(query, key, value, need_weights=False)

        x = self.add_norm_after_attention(attn_output, query)

        ff_output = self.feed_forward(x)
        x = self.add_norm_after_ff(ff_output, x)

        return x


class CustomMambaBlock(nn.Module):
    def __init__(self, d_input, d_model, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_input, d_model)
        self.s_B = nn.Linear(d_model, d_model)
        self.s_C = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_input)
        self.norm = nn.LayerNorm(d_input)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x_in = x
        x = self.in_proj(x)
        B = self.s_B(x)
        C = self.s_C(x)
        x = x + B + C
        x = self.activation(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        x = self.norm(x + x_in)
        return x
