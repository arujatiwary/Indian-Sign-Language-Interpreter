"""Add a --vocab filter to train.py so a subset can be trained without
re-extracting the dataset.

Extraction already cached all 675 sequences. This filters index.csv at load
time, so the 50-class benchmark and the 25-class demo model come from the same
cache.

Run once from the project root:

    python3 patch_train_vocab.py
"""
from pathlib import Path

TARGET = Path("src/train.py")

OLD_ARG = '''    ap.add_argument("--no-motion", action="store_true")'''
NEW_ARG = '''    ap.add_argument("--no-motion", action="store_true")
    ap.add_argument("--vocab", help="text file, one sign per line; trains on "
                                    "only those signs")
    ap.add_argument("--out", default="best.pt",
                    help="checkpoint filename inside checkpoints/")'''

OLD_LOAD = '''    rows = load_index(args.data)
    labels = sorted({r["label"] for r in rows})'''
NEW_LOAD = '''    rows = load_index(args.data)

    if args.vocab:
        wanted = {l.strip().lower().replace(" ", "_")
                  for l in Path(args.vocab).read_text().splitlines()
                  if l.strip() and not l.startswith("#")}
        available = {r["label"] for r in rows}
        missing = wanted - available
        if missing:
            raise SystemExit(
                "these signs are not in the extracted data: "
                + ", ".join(sorted(missing))
                + "\\ncheck spelling against data/labels.txt"
            )
        rows = [r for r in rows if r["label"] in wanted]
        print(f"vocab filter: {len(rows)} videos across {len(wanted)} signs")

    labels = sorted({r["label"] for r in rows})'''

OLD_SAVE = '''            torch.save({"state_dict": model.state_dict(), "labels": labels,
                        "input_dim": input_dim, "use_motion": use_motion},
                       CKPT_DIR / "best.pt")'''
NEW_SAVE = '''            torch.save({"state_dict": model.state_dict(), "labels": labels,
                        "input_dim": input_dim, "use_motion": use_motion},
                       CKPT_DIR / args.out)'''

OLD_LOAD_CKPT = '''    ckpt = torch.load(CKPT_DIR / "best.pt", map_location=device, weights_only=False)'''
NEW_LOAD_CKPT = '''    ckpt = torch.load(CKPT_DIR / args.out, map_location=device, weights_only=False)'''

OLD_METRICS = '''    (CKPT_DIR / "metrics.json").write_text(json.dumps({'''
NEW_METRICS = '''    metrics_name = Path(args.out).stem + "_metrics.json"
    (CKPT_DIR / metrics_name).write_text(json.dumps({'''


def main():
    if not TARGET.exists():
        raise SystemExit(f"{TARGET} not found. run from the project root.")
    text = TARGET.read_text()

    if "--vocab" in text:
        print("already patched, nothing to do")
        return

    edits = [
        (OLD_ARG, NEW_ARG, "argument list"),
        (OLD_LOAD, NEW_LOAD, "index loading"),
        (OLD_SAVE, NEW_SAVE, "checkpoint save"),
        (OLD_LOAD_CKPT, NEW_LOAD_CKPT, "checkpoint reload"),
        (OLD_METRICS, NEW_METRICS, "metrics write"),
    ]
    for old, _, name in edits:
        if old not in text:
            raise SystemExit(f"could not find the {name}; file differs from expected.")
    for old, new, _ in edits:
        text = text.replace(old, new, 1)

    TARGET.write_text(text)
    print(f"applied {len(edits)} edits to {TARGET}")
    print("\ntrain the demo model with:")
    print("  python3 src/train.py --data data --vocab configs/story_vocab.txt "
          "--out story25.pt --epochs 60")


if __name__ == "__main__":
    main()
