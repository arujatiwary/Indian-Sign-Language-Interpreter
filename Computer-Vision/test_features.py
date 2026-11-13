"""Run with: python -m pytest tests -q  (from the project root)"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import FEATURE_DIM, SEQ_LEN  # noqa: E402
from features import add_motion, frame_features, resample  # noqa: E402


def _sample():
    rng = np.random.default_rng(0)
    pose = rng.random((33, 3)).astype(np.float32)
    pose[11], pose[12] = [0.3, 0.4, 0.0], [0.7, 0.4, 0.0]
    return pose, rng.random((21, 3)).astype(np.float32), rng.random((21, 3)).astype(np.float32)


def test_shape_and_flags():
    pose, left, right = _sample()
    assert frame_features(pose, left, right).shape == (FEATURE_DIM,)
    assert list(frame_features(pose, left, right)[-3:]) == [1.0, 1.0, 1.0]
    assert list(frame_features(None, left, None)[-3:]) == [0.0, 1.0, 0.0]


def test_empty_frame_is_zeros():
    v = frame_features(None, None, None)
    assert v.shape == (FEATURE_DIM,) and v.sum() == 0.0


def test_translation_invariant():
    pose, left, right = _sample()
    shift = np.array([0.1, 0.05, 0.0], dtype=np.float32)
    a = frame_features(pose, left, right)
    b = frame_features(pose + shift, left + shift, right + shift)
    assert np.abs(a - b).max() < 1e-5


def test_scale_invariant():
    pose, left, right = _sample()
    c = np.array([0.5, 0.4, 0.0], dtype=np.float32)
    a = frame_features(pose, left, right)
    b = frame_features((pose - c) * 1.7 + c, (left - c) * 1.7 + c, (right - c) * 1.7 + c)
    assert np.abs(a[:153] - b[:153]).max() < 1e-5


def test_resample_lengths():
    for t in (1, 7, 32, 200):
        seq = np.random.rand(t, FEATURE_DIM).astype(np.float32)
        assert resample(seq, SEQ_LEN).shape == (SEQ_LEN, FEATURE_DIM)


def test_motion_doubles_width():
    seq = np.random.rand(SEQ_LEN, FEATURE_DIM).astype(np.float32)
    out = add_motion(seq)
    assert out.shape == (SEQ_LEN, FEATURE_DIM * 2)
    assert np.abs(out[0, FEATURE_DIM:]).max() == 0.0  # first delta is zero
