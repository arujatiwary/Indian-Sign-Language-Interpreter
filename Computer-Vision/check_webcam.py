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
    "enable your terminal (Terminal, iTerm, or VS Code), then fully quit it "
    "with Cmd-Q and reopen. a running process keeps its old permission state.\n"
    "note that a denied permission does not raise an error on macOS. the "
    "camera opens and hands back all-black frames, so 'it opened fine' is not "
    "evidence that it works."
)

BLACK_THRESHOLD = 6.0   # mean pixel value below this is worth a warning, not an abort
WARMUP_READS = 60       # frames to read through before judging brightness (sensor auto-exposure)
WARMUP_SLEEP = 0.05
BRIGHTNESS_SAMPLE = 10  # last N warm-up frames averaged for the brightness report


def probe_camera(idx, reads=WARMUP_READS, sleep=WARMUP_SLEEP, sample=BRIGHTNESS_SAMPLE):
    """Open one index, read through warm-up, and report what it delivered.

    Judges brightness on an average of the last `sample` frames, not the
    first frame -- the first frames off a macOS camera are black while the
    sensor auto-exposes. Returns (cap, frame, brightness) with cap left open
    on success (caller owns release), or (None, None, None) on failure.
    """
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        return None, None, None

    last_frame = None
    brightness_window = []
    for _ in range(reads):
        ok, frame = cap.read()
        if ok:
            last_frame = frame
            brightness_window.append(float(frame.mean()))
            brightness_window = brightness_window[-sample:]
        else:
            time.sleep(sleep)

    if last_frame is None:
        cap.release()
        return None, None, None

    brightness = sum(brightness_window) / len(brightness_window)
    return cap, last_frame, brightness


def list_cameras(limit=5):
    """Probe indices and report which ones actually hand over a frame."""
    print("probing camera indices, this takes a few seconds\n")
    for idx in range(limit):
        cap, frame, brightness = probe_camera(idx)
        if cap is None:
            print(f"  {idx}: not opened, or opened but delivered no frames")
            continue
        h, w = frame.shape[:2]
        note = " (dark -- may be a warm-up fluke or denied permission)" if brightness < BLACK_THRESHOLD else ""
        print(f"  {idx}: OK, {w}x{h}, mean brightness {brightness:.1f}{note}")
        cap.release()
    print(f"\nif nothing delivered frames:\n{PERMISSION_HINT}")


def find_working_camera(limit=5, skip=None):
    """Return the index of the first camera that delivers real frames."""
    for idx in range(limit):
        if idx == skip:
            continue
        cap, frame, brightness = probe_camera(idx)
        if cap is not None:
            return idx, cap, frame, brightness
    return None, None, None, None


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
    ap.add_argument("--camera", type=int, default=0,
                    help="camera index to try first; if it delivers no frames "
                         "at all, other indices are probed automatically")
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

    brightness = None
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"cannot open source {args.video}. {PERMISSION_HINT}")
    else:
        # AVFoundation explicitly. The default backend on macOS sometimes
        # opens a device that never delivers frames. Judge on an average of
        # several warm-up frames, not the first -- the first frames off a
        # macOS camera are black while the sensor auto-exposes, so a single
        # early frame is not evidence either way.
        camera_index = args.camera
        cap, frame, brightness = probe_camera(camera_index)
        chosen = (camera_index, cap, frame, brightness) if cap is not None else None

        # Try other indices when the requested one is a dud (no frames) or
        # dark (frames, but likely the wrong device / stale warm-up). Never
        # abort on this -- only prefer a brighter alternative if one exists.
        if chosen is None or chosen[3] < BLACK_THRESHOLD:
            reason = ("delivered no frames" if chosen is None
                      else f"warm-up frames are dark (mean {chosen[3]:.1f})")
            print(f"camera {camera_index} {reason}; probing other indices...")
            found_index, found_cap, found_frame, found_brightness = find_working_camera(skip=camera_index)
            if found_cap is not None and (chosen is None or found_brightness > chosen[3]):
                if chosen is not None:
                    chosen[1].release()
                print(f"using camera {found_index} instead (mean brightness {found_brightness:.1f})")
                chosen = (found_index, found_cap, found_frame, found_brightness)
            elif found_cap is not None:
                found_cap.release()

        if chosen is None:
            raise SystemExit(
                f"no camera on indices 0-4 delivered frames.\n{PERMISSION_HINT}"
            )

        camera_index, cap, _, brightness = chosen

        if brightness < BLACK_THRESHOLD:
            print(f"warning: camera {camera_index} warm-up frames are dark "
                  f"(mean {brightness:.1f}). continuing anyway -- this can be a warm-up "
                  f"fluke rather than denied permission. if hands/pose are never detected "
                  f"below, that confirms a permissions problem:\n{PERMISSION_HINT}\n")

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
