"""Baseline: PatchTST run from the ORIGINAL Time-Series-Library source.

The upstream file thuml/Time-Series-Library `models/PatchTST.py` (sha 085efd8) lives
byte-for-byte at `models/_tslib/PatchTST.py`. This module only adapts its
`forward(x_enc, x_mark, x_dec, x_mark, mask)` + `task_name` interface to the TimeCAP
encoder's `forward(x_time, x_text) -> (logits, emb)`. No model logic is changed.

TSLib PatchTST's classification path does its own non-stationary (mean/std) normalization
internally, uses PatchEmbedding + Transformer encoder, then flatten -> dropout -> projection.
"""
import torch.nn as nn

from models._tslib.PatchTST import Model as TSLibPatchTST


class Model(nn.Module):
    def __init__(self, configs, device=None):
        super().__init__()
        configs.task_name = 'classification'
        self.net = TSLibPatchTST(configs)

    def forward(self, x_enc_time, x_enc_text=None):
        logits = self.net(x_enc_time, None, None, None)  # (B, num_class)
        return logits, logits
