"""Live recognition: webcam or video file to a stream of sign words.

This is the stage that sits between the trained classifier and the language
model. The classifier wants a fixed 32-frame clip; a camera gives an endless
stream. This file bridges the two.

    python src/realtime.py                        # webcam, prints words
    python src/realtime.py --story                # also generates sentences
    python src/realtime.py --video path.MOV       # replay a dataset clip
    python src/realtime.py --headless             # no preview window

The buffer logic is kept free of torch and mediapipe so it can be unit tested
without a camera or a GPU. Only `SignRecognizer` and `run` touch those.
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import BUFFER_LEN, CKPT_DIR, FEATURE_DIM, SEQ_LEN  # noqa: E402
from features import add_motion, resample  # noqa: E402
from phrase import PhraseConfig, PhraseDetector  # noqa: E402

# Movement below this counts as a pause. Measured, not guessed: run
# demo_video.py --debug, which prints the motion level for every sign region
# and every gap, and pick a value between the two clusters.
STILL_EPS = 0.004


class RollingBuffer:
    """The last N feature vectors, resampled on demand to the model's length.

    Training resampled whole clips to 32 frames, so inference has to hand the
    model something built the same way. The buffer holds 48 frames, about two
    seconds, and resamples that window down to 32. A window shorter than the
    model's input would be padded with meaningless zeros, so the buffer reports
    `ready` only once it holds enough.
    """

    def __init__(self, maxlen=BUFFER_LEN, seq_len=SEQ_LEN, min_fill=None):
        self.frames = deque(maxlen=maxlen)
        self.seq_len = seq_len
        # Predicting on a half-full buffer means predicting on half a sign.
        self.min_fill = min_fill if min_fill is not None else seq_len

    def push(self, vec):
        vec = np.asarray(vec, dtype=np.float32)
        if vec.shape != (FEATURE_DIM,):
            raise ValueError(f"expected ({FEATURE_DIM},), got {vec.shape}")
        self.frames.append(vec)

    @property
    def ready(self):
        return len(self.frames) >= self.min_fill

    @property
    def hands_present(self):
        """True if either hand was detected in the most recent frame."""
        if not self.frames:
            return False
        return bool(self.frames[-1][-2] or self.frames[-1][-1])

    def hands_present_recent(self, n=3):
        """True if a hand appeared in any of the last n frames.

        A single dropped detection mid-sign is common, so a one-frame test
        would end phrases in the middle of them.
        """
        if not self.frames:
            return False
        for vec in list(self.frames)[-n:]:
            if vec[-2] or vec[-1]:
                return True
        return False

    def motion(self, n=6):
        """Mean absolute landmark change per frame over the last n frames.

        Hand presence is the wrong boundary signal on dataset video. The
        presence flag means "a hand was found", not "a hand is raised", and a
        signer standing at rest still has both hands in shot, so a stitched
        pause reads as 100% hands present and no phrase ever ends.

        A pause has little motion, and that is measurable. Note that a repeated
        frame does not give identical landmarks: MediaPipe's video mode carries
        tracking state between frames, so the same pixels still jitter. The
        mean over the whole vector averages that jitter down while real
        movement stays high, which is why this is a mean and not a maximum.

        Returns 0.0 until the window has filled, so an opening frame is never
        mistaken for a pause.
        """
        if len(self.frames) < n + 1:
            return 0.0
        win = np.stack(list(self.frames)[-(n + 1):])
        return float(np.abs(np.diff(win, axis=0)).mean())

    def is_static(self, n=6, eps=STILL_EPS):
        """True if the last n frames show no real movement."""
        if len(self.frames) < n + 1:
            return False
        return self.motion(n) < eps

    def sequence(self, with_motion=True):
        """The window as the model expects it: (seq_len, 282) or (seq_len, 564)."""
        if not self.frames:
            raise ValueError("buffer is empty")
        seq = resample(np.stack(self.frames), self.seq_len)
        return add_motion(seq) if with_motion else seq

    def clear(self):
        self.frames.clear()


class SignRecognizer:
    """Wraps the trained checkpoint. Imports torch lazily so the buffer above
    can be tested in an environment without it."""

    def __init__(self, checkpoint=None, device=None):
        import torch
        from model import SignLSTM

        path = Path(checkpoint) if checkpoint else CKPT_DIR / "best.pt"
        if not path.exists():
            raise SystemExit(f"no checkpoint at {path}. train first.")

        if device is None:
            device = ("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
        self.torch = torch
        self.device = torch.device(device)

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.labels = ckpt["labels"]
        self.use_motion = ckpt.get("use_motion", True)
        self.model = SignLSTM(ckpt["input_dim"], len(self.labels)).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict(self, sequence):
        """(seq_len, D) in, (label, confidence) out."""
        torch = self.torch
        x = torch.from_numpy(np.asarray(sequence, dtype=np.float32))[None]
        with torch.no_grad():
            probs = torch.softmax(self.model(x.to(self.device)), dim=1)[0]
        idx = int(probs.argmax())
        return self.labels[idx], float(probs[idx])


def run(args):
    import cv2
    from extractor import LandmarkExtractor

    recog = SignRecognizer(args.checkpoint)
    print(f"{len(recog.labels)} signs, device {recog.device}")

    buf = RollingBuffer()
    det = PhraseDetector(PhraseConfig(min_confidence=args.confidence))

    teller = None
    if args.story:
        from storyteller import Storyteller
        teller = Storyteller(temperature=args.temperature)
        if teller.provider == "none":
            print("no API key, sentences will use the offline template")

    speak = None
    if args.speak:
        from tts import speak as _speak
        speak = _speak

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {args.video}")
    else:
        cap = _open_camera(cv2, args.camera)

    words, start, i, misses = [], time.time(), 0, 0
    latencies = []

    print("running. press q in the window to stop, or ctrl-c.\n")
    try:
        with LandmarkExtractor(running_mode="video") as ext:
            while True:
                ok, frame = cap.read()
                if not ok:
                    misses += 1
                    if args.video or misses > 60:
                        break
                    time.sleep(0.02)
                    continue
                misses = 0

                if not args.video and (time.time() - start) > args.seconds:
                    break

                ts = i * 40 if args.video else int((time.time() - start) * 1000)
                buf.push(ext.frame(frame, ts))
                i += 1

                label, conf = None, 0.0
                if buf.ready and i % args.predict_every == 0:
                    t0 = time.perf_counter()
                    label, conf = recog.predict(buf.sequence(recog.use_motion))
                    latencies.append((time.perf_counter() - t0) * 1000)

                before = len(det.words)
                phrase = det.update(label, conf, buf.hands_present_recent())
                if len(det.words) > before:
                    print(f"  [{det.words[-1]}]")

                if phrase:
                    words.extend(phrase)
                    print(f"\nphrase: {' '.join(phrase).upper()}")
                    if teller:
                        sentence = teller.add(phrase)
                        print(f"  -> {sentence}")
                        if teller.last_error:
                            print(f"  (template fallback: {teller.last_error})")
                        if speak:
                            speak(sentence)
                    print()

                if not args.headless:
                    _draw(cv2, frame, label, conf, det.words, args.video)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    leftover = det.flush()
    if leftover:
        words.extend(leftover)
        print(f"\nphrase: {' '.join(leftover).upper()}")
        if teller:
            print(f"  -> {teller.add(leftover)}")

    elapsed = time.time() - start
    print(f"\nframes {i} | {i / max(elapsed, 1e-6):.1f} fps")
    if latencies:
        lat = sorted(latencies)
        print(f"inference {np.mean(lat):.1f} ms mean, "
              f"{lat[int(len(lat) * 0.9)]:.1f} ms p90")
    print(f"words: {' '.join(words).upper() if words else '(none)'}")
    if teller and teller.story:
        print(f"\nstory: {teller.story}")


def _open_camera(cv2, index):
    """Same warm-up discipline as check_webcam: early frames are black."""
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {index}. "
                         "run check_webcam.py --list-cameras")
    for _ in range(40):
        ok, frame = cap.read()
        if ok and frame.mean() >= 6.0:
            return cap
        time.sleep(0.03)
    print("warning: camera frames look dark. "
          f"opencv {cv2.__version__}; if 5.x, downgrade below 5.")
    return cap


def _draw(cv2, frame, label, conf, words, is_video):
    view = frame if is_video else cv2.flip(frame, 1)   # extractor got unmirrored
    if label:
        cv2.putText(view, f"{label} {conf:.2f}", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 200, 90), 2, cv2.LINE_AA)
    if words:
        cv2.putText(view, " ".join(words[-6:]).upper(), (12, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 170, 60), 2, cv2.LINE_AA)
    cv2.imshow("sign storyteller", view)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="default: checkpoints/best.pt")
    ap.add_argument("--video", help="replay a file instead of the camera")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--confidence", type=float, default=0.75)
    ap.add_argument("--predict-every", type=int, default=3,
                    help="run the classifier every Nth frame")
    ap.add_argument("--story", action="store_true", help="generate sentences")
    ap.add_argument("--speak", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--headless", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
