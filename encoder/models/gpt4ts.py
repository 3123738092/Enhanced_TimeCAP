import torch
from torch import nn
from transformers import GPT2Model
from layers.Embed import PatchEmbedding


class Model(nn.Module):
    """GPT4TS / One-Fits-All (Zhou et al. 2023): a frozen pretrained GPT-2 backbone applied to
    patched time series, then a class head. Self-attention and FFN are frozen; only the positional
    embeddings and LayerNorm are fine-tuned. Time-series only; the text argument is ignored.
    ``e_layers`` controls how many GPT-2 blocks are kept (paper default 6; TimeCAP searched 1-2)."""

    def __init__(self, configs, device, patch_len=16, stride=8):
        super().__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.device = device
        self.gpt_layers = max(1, int(configs.e_layers))
        padding = stride
        # GPT-2 hidden size is 768; patch embedding projects each patch to 768.
        self.patch_embedding = PatchEmbedding(768, patch_len, stride, padding, configs.dropout)
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        for name, p in self.gpt2.named_parameters():
            p.requires_grad = ('ln' in name) or ('wpe' in name)
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(configs.dropout)
        patch_num = int((configs.seq_len - patch_len) / stride + 2)
        self.projection = nn.Linear(768 * patch_num * configs.enc_in, configs.num_class)

    def forward(self, x_enc_time, x_enc_text=None):
        means = x_enc_time.mean(1, keepdim=True).detach()
        x = x_enc_time - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        x = x.permute(0, 2, 1)                                   # [B, C, L]
        enc_out, n_vars = self.patch_embedding(x)               # [B*C, patch_num, 768]
        out = self.gpt2(inputs_embeds=enc_out).last_hidden_state  # [B*C, patch_num, 768]
        out = torch.reshape(out, (-1, n_vars, out.shape[-2], out.shape[-1]))
        out = self.flatten(out)                                 # [B, C, patch_num*768]
        emb = self.dropout(out.reshape(out.shape[0], -1))
        logits = self.projection(emb)
        return logits, emb
