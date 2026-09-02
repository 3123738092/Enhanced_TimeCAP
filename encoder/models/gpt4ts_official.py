"""Official GPT4TS / One-Fits-All (Zhou et al. 2023) classification model, ported verbatim from
DAMO-DI-ML/NeurIPS2023-One-Fits-All (Classification/src/models/gpt4ts.py + embed.py) into the
TimeCAP encoder interface: Model(configs, device) + forward(x_time, x_text) -> (logits, emb).

Differences vs our simplified models/gpt4ts.py (which is why this matches the paper better):
  - keeps the FULL 6 GPT-2 layers (not 1-2);
  - channel-MIXED patches (all features flattened into each patch token), one sequence per sample;
  - Conv1d value embedding + fixed sinusoidal positional embedding (official DataEmbedding);
  - gelu -> LayerNorm(d_model*patch_num) -> Linear head; no input normalization.
Only fix vs official: dropout is wired to configs.dropout (official had it unused due to an arg bug).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model
from einops import rearrange


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3,
                                   padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


class Model(nn.Module):
    def __init__(self, configs, device, patch_size=16, stride=8):
        super().__init__()
        self.seq_len = configs.seq_len
        self.patch_size = int(getattr(configs, 'patch_size', patch_size))
        self.stride = int(getattr(configs, 'stride', stride))
        self.gpt_layers = 6
        self.feat_dim = configs.enc_in
        self.num_classes = configs.num_class
        self.d_model = 768  # GPT-2 hidden size

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, self.d_model, configs.dropout)

        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        for name, param in self.gpt2.named_parameters():
            param.requires_grad = ('ln' in name) or ('wpe' in name)

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.ln_proj = nn.LayerNorm(self.d_model * self.patch_num)
        self.out_layer = nn.Linear(self.d_model * self.patch_num, self.num_classes)

    def forward(self, x_enc, x_enc_text=None):
        B, L, M = x_enc.shape
        x = rearrange(x_enc, 'b l m -> b m l')
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> b n (p m)')
        x = self.enc_embedding(x)
        x = self.gpt2(inputs_embeds=x).last_hidden_state
        x = self.act(x).reshape(B, -1)
        emb = self.ln_proj(x)
        logits = self.out_layer(emb)
        return logits, emb
