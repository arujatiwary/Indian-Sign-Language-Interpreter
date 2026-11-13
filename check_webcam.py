"""Sanity check the landmark pipeline before running bulk extraction.

    python src/check_webcam.py                    # auto-picks a working camera
    python src/check_webcam.py --camera 1         # force an index
    python src/check_webcam.py --video path.mov   # one file
    python src/check_webcam.py --list-cameras     # probe indices only

Design rule: nothing here exits early on a heuristic. Brightness, detection
rates, and dropped frames are all reported as warnings after the run. The only
fatal errors are a missing model file and a camera that cannot be opened at
all. Everything else lets you see the window and judge for yourself.

Sign a few one-handed and two-handed signs at the camera. What you want:
detection rates near 1.0, and the left/right counts matching the hand you
actually raised.

The preview is mirrored because that feels natural, but the frame handed to
the extractor is not. Mediapipe labels handedness from the image's point of
view, so mirroring before extraction swaps left and right against the
training data.

If capture returns all-black frames on macOS, check the OpenCV version first.
OpenCV 5.0 opens the camera, reports the correct resolution, and delivers
zeros with no error. Pin below 5.
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

DARK = 6.0          # mean pixel value below this is effectively black
WARMUP_READS = 40   # frames to try before giving up on a camera
MAX_MISSES = 60     # consecutive failed reads before calling the stream dead


def open_camera(index):
    """Open a camera and sample it. Returns (cap, brightness) or (None, 0.0).

    Brightness is the brightest frame seen during warm-up, not the first one.
    Early frames are routinely black while the sensor auto-exposes.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None, 0.0

    brightest = 0.0
    got_any = False
    for _ in range(WARMUP_READS):
        ok, frame = cap.read()
        if ok:
            got_any = True
            brightest = max(brightest, float(frame.mean()))
            if brightest >= DARK:
                return cap, brightest
        time.sleep(0.03)

    if got_any:
        return cap, brightest      # opened and delivering, just dark
    cap.release()
    return None, 0.0


def list_cameras(limit=5):
    print("probing camera indices\n")
    found = []
    for idx in range(limit):
        cap, brightness = open_camera(idx)
        if cap is None:
            print(f"  {idx}: no frames")
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        state = "OK" if brightness >= DARK else "DARK"
        print(f"  {idx}: {state}, {w}x{h}, brightness {brightness:.1f}")
        found.append((idx, brightness))
        cap.release()
    return found


def pick_camera(preferred=None):
    """Use the requested index, else the first one delivering a lit frame."""
    if preferred is not None:
        cap, brightness = open_camera(preferred)
        if cap is None:
            raise SystemExit(
                f"camera {preferred} delivered no frames. "
                "run --list-cameras to see what is available."
            )
        return cap, preferred, brightness

    for idx in range(5):
        cap, brightness = open_camera(idx)
        if cap is None:
            continue
        if brightness >= DARK:
            print(f"using camera {idx} (brightness {brightness:.1f})")
            return cap, idx, brightness
        cap.release()

    # nothing lit; fall back to anything that opens at all
    for idx in range(5):
        cap, brightness = open_camera(idx)
        if cap is not None:
            print(f"using camera {idx}, but frames look dark "
                  f"(brightness {brightness:.1f}). continuing anyway.")
            return cap, idx, brightness

    raise SystemExit(
        "no camera delivered frames.\n"
        f"opencv version {cv2.__version__} -- if this is 5.x, that is the "
        "likely cause; run: pip install 'opencv-python<5' 'opencv-contrib-python<5'\n"
        "otherwise check System Settings > Privacy & Security > Camera for the "
        "app running python, then quit it with Cmd-Q and reopen."
    )


def summarise(vectors, elapsed, brightness, misses):
    if not vectors:
        print("\nno frames processed")
        return

    arr = np.stack(vectors)
    pose, left, right = arr[:, -3], arr[:, -2], arr[:, -1]
    n = len(arr)

    print(f"\nframes {n} | {n / max(elapsed, 1e-6):.1f} fps | dropped {misses}")
    print(f"pose detected  {pose.mean():.2f}")
    print(f"left hand      {left.mean():.2f}")
    print(f"right hand     {right.mean():.2f}")
    print(f"both hands     {((left > 0) & (right > 0)).mean():.2f}")
    print(f"no hands       {((left == 0) & (right == 0)).mean():.2f}")
    print(f"feature dim    {arr.shape[1]} (expected {FEATURE_DIM})")

    notes = []
    if brightness < DARK:
        notes.append(
            f"frames were near-black (brightness {brightness:.1f}), so nothing "
            f"could be detected. opencv {cv2.__version__}: if this is 5.x, "
            "downgrade below 5."
        )
    else:
        if pose.mean() < 0.8:
            notes.append("pose often missing. step back so your torso and both "
                         "shoulders are in frame; without shoulders the body "
                         "normalisation falls back to zeros.")
        if (left.mean() + right.mean()) < 0.5:
            notes.append("hands rarely detected. more light, plainer background, "
                         "and keep hands inside the frame.")
    if misses > n * 0.1:
        notes.append(f"{misses} dropped reads. another app may be competing for "
                     "the camera.")

    for note in notes:
        print(f"\n{note}")
    if not notes:
        print("\nlooks good. check that the L/R counts match the hands you "
              "actually raised, then move on to extraction.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="check a file instead of the webcam")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--list-cameras", action="store_true")
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    for task in (HAND_TASK, POSE_TASK):
        if not Path(task).exists():
            raise SystemExit(f"missing {task}. run: bash scripts/download_models.sh")

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {args.video}")
        brightness = float("inf")
    else:
        cap, _, brightness = pick_camera(args.camera)

    vectors, misses = [], 0
    start = time.time()
    print("running. press q in the window to stop early."
          if not args.headless else "running.")

    try:
        with LandmarkExtractor(running_mode="video") as ext:
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    misses += 1
                    if misses > MAX_MISSES:
                        break
                    time.sleep(0.02)
                    continue

                if not args.video and (time.time() - start) > args.seconds:
                    break

                ts = i * 40 if args.video else int((time.time() - start) * 1000)
                vec = ext.frame(frame, ts)     # unmirrored frame to the extractor
                vectors.append(vec)
                i += 1

                if not args.headless:
                    view = frame if args.video else cv2.flip(frame, 1)
                    hands = ("L" if vec[-2] > 0 else "") + ("R" if vec[-1] > 0 else "")
                    label = f"pose={'y' if vec[-3] > 0 else 'n'} hands={hands or '-'}"
                    cv2.putText(view, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 0), 2)
                    cv2.imshow("landmark check", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    summarise(vectors, time.time() - start, brightness, misses)


if __name__ == "__main__":
    main()