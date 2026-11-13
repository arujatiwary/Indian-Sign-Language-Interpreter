"""Central config. Import from here so the training and real-time paths agree."""

from pathlib import Path

# --- paths -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"          # mediapipe .task bundles
DATA_DIR = ROOT / "data"              # extracted .npy sequences
CKPT_DIR = ROOT / "checkpoints"

HAND_TASK = MODELS_DIR / "hand_landmarker.task"
POSE_TASK = MODELS_DIR / "pose_landmarker_lite.task"

# --- feature layout ----------------------------------------------------
# Pose points kept: nose, shoulders, elbows, wrists, hips.
# Indices follow the mediapipe pose topology (33 points).
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]

N_HAND_PTS = 21
N_POSE_PTS = len(POSE_IDX)

# per frame:
#   pose, body-normalised            9 * 3  = 27
#   two hands, body-normalised      42 * 3  = 126
#   two hands, wrist-local shape    42 * 3  = 126
#   presence flags (pose, left, right)       3
FEATURE_DIM = N_POSE_PTS * 3 + N_HAND_PTS * 3 * 2 * 2 + 3  # 282

# --- sequence ----------------------------------------------------------
SEQ_LEN = 32          # every clip is resampled to this many frames
BUFFER_LEN = 48       # real-time rolling buffer, resampled down to SEQ_LEN

# --- training ----------------------------------------------------------
HIDDEN = 128
LAYERS = 2
DROPOUT = 0.3
LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 60
SEED = 42
