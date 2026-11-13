"""Walk an INCLUDE download and cache one .npy of landmark features per video.

Extraction is the slow part of this project (mediapipe runs on CPU, roughly
real time per video). Doing it once and caching means training runs take
seconds afterwards.

    python src/extract_dataset.py --include-dir ~/Datasets/INCLUDE \
        --out data/ --vocab configs/vocab.txt

Label rule: the video's parent folder name is the sign. INCLUDE names those
folders like "1. loud", so the leading number is stripped.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import SEQ_LEN  # noqa: E402
from extractor import LandmarkExtractor  # noqa: E402
from features import resample  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".MOV", ".MP4"}
LEADING_NUM = re.compile(r"^\s*\d+\s*[.\-_)]\s*")


def clean_label(name):
    return LEADING_NUM.sub("", name).strip().lower().replace(" ", "_")


def find_videos(root, vocab=None):
    items = []
    for path in sorted(Path(root).rglob("*")):
        if path.suffix not in VIDEO_EXT or not path.is_file():
            continue
        label = clean_label(path.parent.name)
        if vocab and label not in vocab:
            continue
        items.append((path, label))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-dir", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--vocab", help="text file, one sign per line; omit to take everything")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--list-only", action="store_true",
                    help="print the sign labels present in the download and exit")
    ap.add_argument("--keep-raw-length", action="store_true",
                    help="cache full-length sequences instead of resampling to SEQ_LEN")
    args = ap.parse_args()

    vocab = None
    if args.vocab:
        vocab = {clean_label(l) for l in Path(args.vocab).read_text().splitlines() if l.strip()}

    videos = find_videos(args.include_dir, vocab)
    if not videos:
        raise SystemExit("no videos matched. check --include-dir and --vocab")

    labels = sorted({lab for _, lab in videos})
    print(f"{len(videos)} videos across {len(labels)} signs")

    if args.list_only:
        counts = {}
        for _, lab in videos:
            counts[lab] = counts.get(lab, 0) + 1
        for lab in sorted(counts, key=lambda k: (-counts[k], k)):
            print(f"{counts[lab]:4d}  {lab}")
        return

    out = Path(args.out)
    (out / "seqs").mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    with LandmarkExtractor(running_mode="video") as ext:
        for i, (path, label) in enumerate(tqdm(videos, desc="extracting")):
            try:
                seq = ext.video(path, stride=args.stride)
            except Exception as e:  # a few INCLUDE files are unreadable
                failures.append((str(path), repr(e)))
                continue
            if not args.keep_raw_length:
                seq = resample(seq, SEQ_LEN)
            name = f"{i:05d}_{label}.npy"
            np.save(out / "seqs" / name, seq)
            hand_frames = float((seq[:, -2:] > 0).any(axis=1).mean())
            rows.append({"file": name, "label": label, "src": str(path),
                         "frames": seq.shape[0], "hand_rate": round(hand_frames, 3)})

    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "src", "frames", "hand_rate"])
        w.writeheader()
        w.writerows(rows)
    (out / "labels.txt").write_text("\n".join(sorted({r["label"] for r in rows})))

    print(f"wrote {len(rows)} sequences to {out/'seqs'}")
    if failures:
        (out / "failures.txt").write_text("\n".join(f"{p}\t{e}" for p, e in failures))
        print(f"{len(failures)} videos failed, see {out/'failures.txt'}")

    weak = [r for r in rows if r["hand_rate"] < 0.5]
    if weak:
        print(f"warning: {len(weak)} clips have hands visible in under half the frames. "
              "inspect these before training, they are usually detection failures.")


if __name__ == "__main__":
    main()
