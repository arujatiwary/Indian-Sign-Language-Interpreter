"""Sanity check the landmark pipeline before running bulk extraction.

    python src/check_webcam.py                    # webcam, press q to stop
    python src/check_webcam.py --video path.mov   # one file
    python src/check_webcam.py --headless         # no window, prints only

Sign a few one-handed and two-handed signs at the camera. What you want to see:
detection rates near 1.0, and the left/right counts matching the hand you
actually raised.

The preview window is mirrored because that is what feels natural, but the
frame handed to the extractor is not. Mediapipe labels handedness from the
image's point of view, so mirroring before extraction swaps left and right
against the training data.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import FEATURE_DIM, HAND_TASK, POSE_TASK  # noqa: E402
from extractor import LandmarkExtractor  # noqa: E402


PERMISSION_HINT = (
    "on macOS, camera access is granted to the app running python, not to "
    "python itself. open System Settings > Privacy & Security > Camera and "
    "enable your terminal (Terminal, iTerm, or VS Code), then fully quit and "
    "reopen it. the prompt only appears once, so if you dismissed it you must "
    "enable it by hand."
)


def list_cameras(limit=5):
    """Probe indices and report which ones actually hand over a frame."""
    print("probing camera indices, this takes a few seconds\n")
    for idx in range(limit):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print(f"  {idx}: not opened")
            cap.release()
            continue
        got = False
        for _ in range(20):
            ok, frame = cap.read()
            if ok:
                got = True
                break
            time.sleep(0.05)
        if got:
            h, w = frame.shape[:2]
            print(f"  {idx}: OK, {w}x{h}")
        else:
            print(f"  {idx}: opened but no frames (permission, or in use by another app)")
        cap.release()
    print(f"\nif nothing delivered frames:\n{PERMISSION_HINT}")


def summarise(vectors, elapsed):
    if not vectors:
        print("no frames processed")
        return
    arr = np.stack(vectors)
    pose, left, right = arr[:, -3], arr[:, -2], arr[:, -1]
    n = len(arr)
    print(f"\nframes {n} | {n / max(elapsed, 1e-6):.1f} fps")
    print(f"pose detected  {pose.mean():.2f}")
    print(f"left hand      {left.mean():.2f}")
    print(f"right hand     {right.mean():.2f}")
    print(f"both hands     {((left > 0) & (right > 0)).mean():.2f}")
    print(f"no hands       {((left == 0) & (right == 0)).mean():.2f}")
    print(f"feature dim    {arr.shape[1]} (expected {FEATURE_DIM})")

    if pose.mean() < 0.8:
        print("\npose is missing often. step back so your torso and both "
              "shoulders are in frame. without shoulders the body "
              "normalisation falls back to zeros.")
    if (left.mean() + right.mean()) < 0.5:
        print("\nhands are rarely detected. try more light and a plainer "
              "background.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="check a file instead of the webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--list-cameras", action="store_true",
                    help="probe camera indices 0-4 and report which deliver frames")
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    for task in (HAND_TASK, POSE_TASK):
        if not Path(task).exists():
            raise SystemExit(f"missing {task}. run: bash scripts/download_models.sh")

    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        # AVFoundation explicitly. The default backend on macOS sometimes
        # opens a device that never delivers frames.
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        raise SystemExit(f"cannot open source {args.video or args.camera}. {PERMISSION_HINT}")

    if not args.video:
        # macOS cameras take a moment to start delivering. isOpened() is true
        # well before the first frame arrives, and it is also true when
        # permission is denied, so a warm-up read is the real test.
        for attempt in range(60):
            ok, _ = cap.read()
            if ok:
                break
            time.sleep(0.05)
        else:
            cap.release()
            raise SystemExit(
                f"camera {args.camera} opened but delivered no frames after 3s.\n"
                f"{PERMISSION_HINT}\n"
                "if permission is already granted, run "
                "'python src/check_webcam.py --list-cameras' to find the right index."
            )

    vectors = []
    start = time.time()
    print("running. press q in the window to stop early." if not args.headless else "running.")

    with LandmarkExtractor(running_mode="video") as ext:
        i = 0
        misses = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                misses += 1
                if misses > 30:      # a real end of stream, or the camera died
                    break
                time.sleep(0.02)
                continue
            misses = 0
            if not args.video and (time.time() - start) > args.seconds:
                break

            # unmirrored frame to the extractor
            vec = ext.frame(frame, int((time.time() - start) * 1000) if not args.video else i * 40)
            vectors.append(vec)
            i += 1

            if not args.headless:
                view = cv2.flip(frame, 1) if not args.video else frame
                hands = []
                if vec[-2] > 0:
                    hands.append("L")
                if vec[-1] > 0:
                    hands.append("R")
                label = f"pose={'y' if vec[-3] > 0 else 'n'} hands={''.join(hands) or '-'}"
                cv2.putText(view, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)
                cv2.imshow("landmark check", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()
    summarise(vectors, time.time() - start)


if __name__ == "__main__":
    main()
