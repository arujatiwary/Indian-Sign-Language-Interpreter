"""Draw detected landmarks over frames from a dataset video.

    python src/make_overlay.py --video "data/INCLUDE/Places/24. School/MVI_3458.MOV"
    python src/make_overlay.py --video ... --frames 6 --out figures

Writes overlay_<sign>.png: a strip of frames with the hand and pose skeleton
drawn on, plus a second figure showing the 282-float feature sequence as a
heatmap. Together they show the whole vision front end in two images: what
MediaPipe sees, and what the LSTM actually receives.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import BaseOptions  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

from config import HAND_TASK, POSE_TASK, SEQ_LEN  # noqa: E402
from extractor import LandmarkExtractor  # noqa: E402
from features import resample  # noqa: E402

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (11, 23), (12, 24), (23, 24)]

# OpenCV uses BGR, so these read backwards from the usual hex order.
LEFT = (255, 160, 60)      # blue
RIGHT = (90, 200, 90)      # green
BODY = (60, 130, 220)      # orange


def draw(frame, hand_res, pose_res):
    h, w = frame.shape[:2]
    out = frame.copy()

    if pose_res.pose_landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in pose_res.pose_landmarks[0]]
        for a, b in POSE_EDGES:
            cv2.line(out, pts[a], pts[b], BODY, 2, cv2.LINE_AA)
        for i in (0, 11, 12, 13, 14, 15, 16, 23, 24):
            cv2.circle(out, pts[i], 4, BODY, -1, cv2.LINE_AA)

    for i, hand in enumerate(hand_res.hand_landmarks):
        label = hand_res.handedness[i][0].category_name
        colour = LEFT if label == "Left" else RIGHT
        pts = [(int(p.x * w), int(p.y * h)) for p in hand]
        for a, b in HAND_EDGES:
            cv2.line(out, pts[a], pts[b], colour, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(out, p, 3, colour, -1, cv2.LINE_AA)
        cv2.putText(out, label, (pts[0][0] - 20, pts[0][1] + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"{video} not found")
    for task in (HAND_TASK, POSE_TASK):
        if not Path(task).exists():
            raise SystemExit(f"missing {task}. run scripts/download_models.sh")

    sign = video.parent.name
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"no frames read from {video}")
    print(f"{len(frames)} frames from {video.name}")

    hands = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_TASK)),
            running_mode=vision.RunningMode.IMAGE, num_hands=2))
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_TASK)),
            running_mode=vision.RunningMode.IMAGE, num_poses=1))

    # evenly spaced frames, skipping the very start and end where the signer
    # is usually still
    idx = np.linspace(len(frames) * 0.15, len(frames) * 0.85,
                      args.frames).astype(int)

    panels = []
    for i in idx:
        frame = frames[i]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(rgb))
        panels.append((i, draw(frame, hands.detect(image), pose.detect(image))))
    hands.close()
    pose.close()

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(3.1 * len(panels), 4.4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (i, img) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"frame {i}", fontsize=9, color="#1a1a1a")
        ax.axis("off")
    fig.suptitle(f'MediaPipe landmarks on "{sign}"', fontsize=13, y=0.97)
    fig.tight_layout()
    name = out / f"overlay_{sign.replace(' ', '_').replace('.', '')}.png"
    fig.savefig(name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name.name}")

    # what the classifier actually receives
    with LandmarkExtractor(running_mode="video") as ext:
        seq = ext.video(video)
    seq = resample(seq, SEQ_LEN)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(seq.T, aspect="auto", cmap="RdBu_r",
                   vmin=-2.5, vmax=2.5, interpolation="nearest")
    for y, tag in [(0, "pose"), (27, "left hand"), (90, "right hand"),
                   (153, "left shape"), (216, "right shape"), (279, "flags")]:
        ax.axhline(y, color="#1a1a1a", linewidth=0.7, alpha=0.6)
        ax.text(SEQ_LEN + 0.4, y + 12, tag, fontsize=8, va="center",
                color="#1a1a1a")
    ax.set_xlabel("frame (resampled to 32)")
    ax.set_ylabel("feature index")
    ax.set_title(f'Feature sequence for "{sign}": {SEQ_LEN} x {seq.shape[1]}')
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.09,
                 label="normalised value")
    fig.tight_layout()
    name = out / f"features_{sign.replace(' ', '_').replace('.', '')}.png"
    fig.savefig(name, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name.name}")


if __name__ == "__main__":
    main()