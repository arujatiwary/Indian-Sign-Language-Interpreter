"""Turn raw landmarks into a normalised feature vector.

No mediapipe import here on purpose. This module is pure numpy so it can be
tested on its own, and so the training path and the webcam path share exactly
the same maths.

Frame layout (282 floats):
    [0:27]     9 pose points, body-normalised
    [27:90]    left hand, body-normalised
    [90:153]   right hand, body-normalised
    [153:216]  left hand, wrist-local shape
    [216:279]  right hand, wrist-local shape
    [279:282]  presence flags: pose, left hand, right hand
"""

import numpy as np

from config import FEATURE_DIM, N_HAND_PTS, POSE_IDX

EPS = 1e-6

L_SHOULDER, R_SHOULDER = 11, 12
WRIST, MIDDLE_MCP = 0, 9


def _body_frame(pose):
    """Origin at the shoulder midpoint, scale = shoulder width.

    Both are read off the same frame, so the features stop depending on how far
    the signer stands from the camera or where they stand in it. This is what
    lets a model trained on INCLUDE video work on your webcam.
    """
    if pose is None:
        return None, None
    left, right = pose[L_SHOULDER], pose[R_SHOULDER]
    origin = (left + right) / 2.0
    scale = float(np.linalg.norm(left[:2] - right[:2]))
    if scale < EPS:
        return None, None
    return origin, scale


def _hand_shape(hand):
    """Wrist-centred, scaled by palm length. Isolates handshape from arm position."""
    if hand is None:
        return np.zeros((N_HAND_PTS, 3), dtype=np.float32)
    centred = hand - hand[WRIST]
    scale = float(np.linalg.norm(centred[MIDDLE_MCP][:2]))
    if scale < EPS:
        scale = 1.0
    return (centred / scale).astype(np.float32)


def frame_features(pose=None, left=None, right=None):
    """Build one frame vector.

    pose  : (33, 3) array or None
    left  : (21, 3) array or None
    right : (21, 3) array or None
    """
    origin, scale = _body_frame(pose)
    has_pose = origin is not None

    def to_body(points):
        if points is None or not has_pose:
            return np.zeros((len(points) if points is not None else 0, 3), dtype=np.float32)
        return ((points - origin) / scale).astype(np.float32)

    pose_block = (
        to_body(pose[POSE_IDX]) if has_pose else np.zeros((len(POSE_IDX), 3), dtype=np.float32)
    )
    left_body = to_body(left) if left is not None else np.zeros((N_HAND_PTS, 3), dtype=np.float32)
    right_body = to_body(right) if right is not None else np.zeros((N_HAND_PTS, 3), dtype=np.float32)
    if left_body.shape[0] == 0:
        left_body = np.zeros((N_HAND_PTS, 3), dtype=np.float32)
    if right_body.shape[0] == 0:
        right_body = np.zeros((N_HAND_PTS, 3), dtype=np.float32)

    flags = np.array(
        [float(has_pose), float(left is not None), float(right is not None)], dtype=np.float32
    )

    vec = np.concatenate(
        [
            pose_block.ravel(),
            left_body.ravel(),
            right_body.ravel(),
            _hand_shape(left).ravel(),
            _hand_shape(right).ravel(),
            flags,
        ]
    ).astype(np.float32)

    assert vec.shape[0] == FEATURE_DIM, f"got {vec.shape[0]}, expected {FEATURE_DIM}"
    return vec


def resample(seq, length):
    """Resample a (T, D) sequence to (length, D) by uniform index sampling.

    INCLUDE clips vary from under a second to several seconds. Resampling keeps
    the whole gesture instead of cutting it off the way padding or truncating
    would, at the cost of losing absolute speed.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError(f"expected (T, D), got {seq.shape}")
    if seq.shape[0] == 0:
        return np.zeros((length, seq.shape[1]), dtype=np.float32)
    idx = np.linspace(0, seq.shape[0] - 1, num=length)
    return seq[np.round(idx).astype(int)]


def add_motion(seq):
    """Append frame-to-frame deltas. Doubles D. Movement is most of ISL."""
    seq = np.asarray(seq, dtype=np.float32)
    delta = np.diff(seq, axis=0, prepend=seq[:1])
    return np.concatenate([seq, delta], axis=1)
