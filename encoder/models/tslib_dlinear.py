"""Baseline: DLinear run from the ORIGINAL Time-Series-Library source.

The upstream file thuml/Time-Series-Library `models/DLinear.py` (sha 3f4d666) lives
byte-for-byte at `models/_tslib/DLinear.py`. This module only adapts its
`forward(x_enc, x_mark, x_dec, x_mark, mask)` + `task_name` interface to the TimeCAP
encoder's `forward(x_time, x_text) -> (logits, emb)`. No model logic is changed.

Note: TSLib DLinear's classification head is `decomp -> Linear_Seasonal/Trend -> projection`,
with NO input normalization and NO dropout (so `--dropout` is inert for this model).
"""
import torch.nn as nn

from models._tslib.DLinear import Model as TSLibDLinear


class Model(nn.Module):
    def __init__(self, configs, device=None):
        super().__init__()
        configs.task_name = 'classification'
        self.net = TSLibDLinear(configs)

    def forward(self, x_enc_time, x_enc_text=None):
        logits = self.net(x_enc_time, None, None, None)  # (B, num_class)
        return logits, logits
