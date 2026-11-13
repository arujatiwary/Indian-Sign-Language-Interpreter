"""Make landmark timestamps monotonic across the lifetime of an extractor.

MediaPipe's VIDEO running mode rejects any frame whose timestamp is not
strictly greater than the previous one seen by that landmarker. Because
extract_dataset.py reuses one extractor for every video, and video() restarts
its clock at zero per file, only the first video is accepted.

Clamping inside frame() fixes it in one place, for every caller.

Run once from the project root:

    python3 patch_extractor_timestamps.py
"""
from pathlib import Path

TARGET = Path("src/extractor.py")

OLD_INIT = '''    def close(self):'''
NEW_INIT = '''        # MediaPipe rejects a timestamp that does not exceed the previous one
        # seen by this landmarker. Each video restarts its own clock at zero,
        # so the extractor keeps its own monotonic counter.
        self._last_ts = -1

    def close(self):'''

OLD_FRAME = '''        """One BGR frame in, one (282,) feature vector out."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)'''
NEW_FRAME = '''        """One BGR frame in, one (282,) feature vector out."""
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)'''


def main():
    if not TARGET.exists():
        raise SystemExit(f"{TARGET} not found. run from the project root.")
    text = TARGET.read_text()

    if "_last_ts" in text:
        print("already patched, nothing to do")
        return
    for old, name in [(OLD_INIT, "close method"), (OLD_FRAME, "frame method")]:
        if old not in text:
            raise SystemExit(f"could not find the {name}; file differs from expected.")

    text = text.replace(OLD_INIT, NEW_INIT, 1).replace(OLD_FRAME, NEW_FRAME, 1)
    TARGET.write_text(text)
    print("patched src/extractor.py")
    print("now delete the stale output and re-run extraction:")
    print("  rm -rf data/seqs data/index.csv data/labels.txt data/failures.txt")


if __name__ == "__main__":
    main()
