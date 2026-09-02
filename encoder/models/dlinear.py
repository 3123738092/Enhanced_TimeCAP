import torch
import torch.nn as nn


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):  # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size - 1 - (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        mean = self.moving_avg(x)
        return x - mean, mean  # seasonal, trend


class Model(nn.Module):
    """DLinear (Zeng et al. 2023) adapted to classification: series decomposition + per-component
    linear over time, then a class head. Time-series only; the text argument is ignored."""

    def __init__(self, configs, device):
        super().__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        k = int(getattr(configs, 'moving_avg', 25))
        if k % 2 == 0:
            k += 1
        self.decomp = series_decomp(k)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.seq_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.seq_len)
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(self.enc_in * self.seq_len, configs.num_class)

    def forward(self, x_enc_time, x_enc_text=None):
        # Non-stationary normalization (same input handling as the other models)
        means = x_enc_time.mean(1, keepdim=True).detach()
        x_enc_time = x_enc_time - means
        stdev = torch.sqrt(torch.var(x_enc_time, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc_time = x_enc_time / stdev
        seasonal, trend = self.decomp(x_enc_time)                    # [B, L, C]
        seasonal = self.Linear_Seasonal(seasonal.permute(0, 2, 1))   # [B, C, L]
        trend = self.Linear_Trend(trend.permute(0, 2, 1))            # [B, C, L]
        out = seasonal + trend                                       # [B, C, L]
        emb = self.dropout(out.reshape(out.shape[0], -1))            # [B, C*L]
        logits = self.projection(emb)
        return logits, emb
