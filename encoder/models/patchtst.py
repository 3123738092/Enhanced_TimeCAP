import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Model(nn.Module):
    """PatchTST (Nie et al. 2023) adapted to classification: patching + vanilla Transformer encoder,
    channel-independent, then a class head. This is the TimeCAP encoder WITHOUT the text branch.
    Time-series only; the text argument is ignored."""

    def __init__(self, configs, device, patch_len=16, stride=8):
        super().__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.device = device
        padding = stride
        self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=configs.output_attention),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
            ) for _ in range(configs.e_layers)],
            norm_layer=torch.nn.LayerNorm(configs.d_model))
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(configs.dropout)
        patch_num = int((configs.seq_len - patch_len) / stride + 2)   # same count as TimeCAP encoder (minus text token)
        self.projection = nn.Linear(configs.d_model * patch_num * configs.enc_in, configs.num_class)

    def forward(self, x_enc_time, x_enc_text=None):
        # Non-stationary normalization (same as TimeCAP encoder)
        means = x_enc_time.mean(1, keepdim=True).detach()
        x = x_enc_time - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        x = x.permute(0, 2, 1)                                   # [B, C, L]
        enc_out, n_vars = self.patch_embedding(x)               # [B*C, patch_num, d_model]
        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)                    # [B, C, d_model, patch_num]
        out = self.flatten(enc_out)                             # [B, C, d_model*patch_num]
        emb = self.dropout(out.reshape(out.shape[0], -1))
        logits = self.projection(emb)
        return logits, emb
