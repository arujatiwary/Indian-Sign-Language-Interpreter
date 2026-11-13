"""BiLSTM sign classifier.

Input  : (B, T, D) landmark features
Output : (B, n_classes) logits

Pooling takes the mean and the max over time rather than the last hidden
state. A sign can peak anywhere in the clip, and the last frame is often the
hand already dropping.
"""

import torch
import torch.nn as nn

from config import DROPOUT, HIDDEN, LAYERS


class SignLSTM(nn.Module):
    def __init__(self, input_dim, n_classes, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        # Standardisation uses fixed training-set statistics rather than
        # LayerNorm. LayerNorm rescales every frame on its own, which would
        # undo the body normalisation the features depend on. These are
        # buffers, so they travel with the checkpoint and the webcam path
        # normalises the same way.
        self.register_buffer("feat_mean", torch.zeros(input_dim))
        self.register_buffer("feat_std", torch.ones(input_dim))
        self.proj = nn.Linear(input_dim, hidden)
        self.lstm = nn.LSTM(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def set_stats(self, mean, std):
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-3))

    def forward(self, x):
        x = (x - self.feat_mean) / self.feat_std
        x = self.proj(x)
        out, _ = self.lstm(x)
        pooled = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        return self.head(pooled)
