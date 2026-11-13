"""Dataset over the cached .npy sequences, with a stratified split."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import SEQ_LEN
from features import add_motion, resample


def load_index(data_dir):
    with open(Path(data_dir) / "index.csv") as f:
        return list(csv.DictReader(f))


def stratified_split(rows, val_frac=0.15, test_frac=0.15, seed=42):
    """Split per class so every sign appears in every split."""
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for label, group in sorted(by_label.items()):
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n_test + n_val >= n:  # tiny class, keep at least one for training
            n_test, n_val = 1, 1
        test += group[:n_test]
        val += group[n_test:n_test + n_val]
        train += group[n_test + n_val:]
    return train, val, test


class SignSequences(Dataset):
    def __init__(self, rows, data_dir, labels, train=False, use_motion=True):
        self.rows = rows
        self.dir = Path(data_dir) / "seqs"
        self.label_to_idx = {l: i for i, l in enumerate(labels)}
        self.train = train
        self.use_motion = use_motion

    def __len__(self):
        return len(self.rows)

    def _augment(self, seq, rng):
        # small rotation in the image plane
        theta = rng.uniform(-0.12, 0.12)
        c, s = np.cos(theta), np.sin(theta)
        pts = seq[:, :279].reshape(seq.shape[0], -1, 3)
        x, y = pts[..., 0].copy(), pts[..., 1].copy()
        pts[..., 0] = c * x - s * y
        pts[..., 1] = s * x + c * y
        seq[:, :279] = pts.reshape(seq.shape[0], -1)

        seq[:, :279] *= rng.uniform(0.9, 1.1)          # scale
        seq[:, :279] += rng.normal(0, 0.01, seq[:, :279].shape)  # jitter

        # drop a short run of frames, mimicking a detection dropout
        if rng.random() < 0.3:
            start = rng.integers(0, max(1, seq.shape[0] - 4))
            seq[start:start + int(rng.integers(1, 4))] = 0.0
        return seq

    def __getitem__(self, i):
        row = self.rows[i]
        seq = np.load(self.dir / row["file"]).astype(np.float32)
        if seq.shape[0] != SEQ_LEN:
            seq = resample(seq, SEQ_LEN)
        if self.train:
            seq = self._augment(seq, np.random.default_rng())
        if self.use_motion:
            seq = add_motion(seq)
        return torch.from_numpy(seq), self.label_to_idx[row["label"]]
