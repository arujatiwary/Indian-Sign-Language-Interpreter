"""K-fold cross-validation with leakage control.

Two separate problems with the single split, and this fixes both.

1. SAMPLE SIZE. The single split leaves about 2 test clips per class, so a
   per-class F1 of 0.00 can mean two unlucky guesses and 1.00 two lucky ones.
   K-fold gives every clip one turn in test, so per-class numbers rest on ~13
   clips, and accuracy comes with a spread rather than a single point.

2. LEAKAGE, which matters more. INCLUDE was recorded by 7 signers who each
   performed every sign 2-3 times in a row. Those repeat takes are near
   duplicates: same person, same sign, seconds apart. Splitting them across
   train and test lets the model recognise a specific person's rendition
   rather than the sign. In the current random split, 82% of test clips have
   a sibling take sitting in training.

   `--group-by run` keeps every take of one recording run inside one fold, so
   a clip is scored only against signers and sittings the model never saw.
   This approximates signer-independent evaluation, which Sabir et al.
   (Findings of EMNLP 2025) argue should be the default for sign language,
   having measured BLEU-4 falling by more than half when the same models are
   re-evaluated with signers held out.

   Run groups are inferred from the camera's sequential file numbering
   (MVI_xxxx): consecutive numbers within a sign folder are one sitting. This
   is a proxy, not a published signer label, because INCLUDE ships no signer
   column. It recovers a median of 7 runs per sign, matching the 7 signers the
   dataset paper reports. Report it as leakage-controlled, not as true
   signer-independent.

    python src/cross_validate.py --folds 5                   # grouped (default)
    python src/cross_validate.py --folds 5 --group-by none   # ungrouped, for the comparison
    python src/cross_validate.py --folds 5 --vocab configs/story_vocab.txt

Report both numbers. The gap between them is itself a result.

Writes checkpoints/cv_metrics.json. Roughly K times a single training run.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from config import BATCH_SIZE, CKPT_DIR, EPOCHS, FEATURE_DIM, LR, SEED  # noqa: E402
from dataset import SignSequences, load_index  # noqa: E402
from model import SignLSTM  # noqa: E402
from train import evaluate, per_class_f1, pick_device  # noqa: E402


def recording_runs(rows, max_gap=5):
    """Label each clip with the recording run it belongs to.

    Cameras number files sequentially, so takes recorded one after another
    carry consecutive numbers. Within a sign folder, a break in the numbering
    marks a new sitting, and in this dataset a new signer.

    Returns {row_id: "label#k"}. Clips with no parsable number each become
    their own run, which is the conservative choice: it never merges two
    unrelated clips.
    """
    nums = {}
    for r in rows:
        m = re.search(r"MVI_(\d+)", r["src"])
        nums[r["file"]] = int(m.group(1)) if m else None

    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    run_of, orphan = {}, 0
    for label, group in by_label.items():
        numbered = sorted((nums[r["file"]], r) for r in group
                          if nums[r["file"]] is not None)
        rid, prev = 0, None
        for n, r in numbered:
            if prev is not None and n - prev > max_gap:
                rid += 1
            run_of[r["file"]] = f"{label}#{rid}"
            prev = n
        for r in group:
            if nums[r["file"]] is None:
                run_of[r["file"]] = f"{label}#orphan{orphan}"
                orphan += 1
    return run_of


def grouped_folds(rows, k, seed=SEED):
    """Stratified folds that never split a recording run.

    Whole runs are dealt to folds. Two things are balanced at once: each
    class is spread as evenly as the run structure allows, and the folds are
    kept similar in total size. Tracking only the per-class count lets the
    global imbalance accumulate across classes, which produced folds of 171
    and 81 clips; carrying the global load as the tie-break fixes it.

    A class with fewer runs than k cannot appear in every fold. That is a
    property of the data, not a bug: the pooled confusion matrix is still
    exact, because every clip is tested exactly once. Only the fold-to-fold
    spread is affected, and the caller is warned.
    """
    run_of = recording_runs(rows)

    runs = defaultdict(list)
    for r in rows:
        runs[run_of[r["file"]]].append(r)

    by_label = defaultdict(list)
    for members in runs.values():
        by_label[members[0]["label"]].append(members)

    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    load = [0] * k                      # carried across classes

    # Rarest classes first: they have the fewest runs and so the least
    # freedom, and placing them before the common ones keeps them spread.
    for label in sorted(by_label, key=lambda l: (len(by_label[l]), l)):
        groups = by_label[label]
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        here = [0] * k
        for g in groups:
            target = min(range(k), key=lambda i: (here[i], load[i], i))
            folds[target].extend(g)
            here[target] += len(g)
            load[target] += len(g)
    return folds


def max_usable_folds(rows):
    """The largest k where every class can still reach every fold."""
    run_of = recording_runs(rows)
    per_class = defaultdict(set)
    for r in rows:
        per_class[r["label"]].add(run_of[r["file"]])
    return min(len(v) for v in per_class.values())


def stratified_folds(rows, k, seed=SEED):
    """Assign each row a fold, balanced per class.

    Classes here hold as few as 7 videos, so a plain random partition can leave
    a class absent from a fold entirely. Dealing each class round-robin into
    folds keeps every class present everywhere.
    """
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        for i, row in enumerate(group):
            folds[i % k].append(row)
    return folds


def train_fold(train_rows, test_rows, labels, data_dir, epochs, device):
    use_motion = True
    input_dim = FEATURE_DIM * 2

    tr = DataLoader(SignSequences(train_rows, data_dir, labels, train=True,
                                  use_motion=use_motion),
                    batch_size=BATCH_SIZE, shuffle=True)
    te = DataLoader(SignSequences(test_rows, data_dir, labels, train=False,
                                  use_motion=use_motion),
                    batch_size=BATCH_SIZE)

    stack = np.concatenate([tr.dataset[i][0].numpy()
                            for i in range(len(train_rows))])
    model = SignLSTM(input_dim, len(labels)).to(device)
    model.set_stats(stack.mean(axis=0), stack.std(axis=0))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    for _ in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    _, acc, confusion = evaluate(model, te, device, len(labels))
    return acc, confusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--vocab")
    ap.add_argument("--group-by", choices=["run", "none"], default="run",
                    help="run: keep each recording run whole, so repeat takes "
                         "by one signer never straddle train and test "
                         "(default). none: the old ungrouped split, for the "
                         "comparison number only")
    ap.add_argument("--out", default="checkpoints/cv_metrics.json")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    rows = load_index(args.data)
    if args.vocab:
        wanted = {l.strip().lower().replace(" ", "_")
                  for l in Path(args.vocab).read_text().splitlines()
                  if l.strip() and not l.startswith("#")}
        rows = [r for r in rows if r["label"] in wanted]

    labels = sorted({r["label"] for r in rows})
    grouped = args.group_by == "run"
    folds = (grouped_folds(rows, args.folds) if grouped
             else stratified_folds(rows, args.folds))
    device = pick_device()

    print(f"{len(rows)} videos | {len(labels)} signs | {args.folds} folds | "
          f"{device}")
    print(f"grouping: {'recording run (leakage controlled)' if grouped else 'NONE (clips split freely)'}")
    print(f"fold sizes: {[len(f) for f in folds]}")

    # State the leakage plainly, whichever mode is running.
    run_of = recording_runs(rows)
    fold_runs = [{run_of[r["file"]] for r in f} for f in folds]
    straddling = sum(
        1 for i in range(args.folds)
        for run in fold_runs[i]
        if any(run in fold_runs[j] for j in range(args.folds) if j != i)
    )
    total_runs = len({run_of[r["file"]] for r in rows})
    print(f"recording runs: {total_runs}  |  runs straddling a fold boundary: "
          f"{straddling}")
    if grouped:
        usable = max_usable_folds(rows)
        if args.folds > usable:
            print(f"note: the rarest class has only {usable} recording runs, so "
                  f"it cannot appear in all {args.folds} folds.")
            print("      pooled per-class metrics are still exact (every clip "
                  "is tested once);")
            print("      only the fold-to-fold spread is affected. "
                  f"--folds {usable} avoids it entirely.")
    print()

    accs, total_confusion = [], np.zeros((len(labels), len(labels)), dtype=int)

    for i in range(args.folds):
        test_rows = folds[i]
        train_rows = [r for j, f in enumerate(folds) if j != i for r in f]
        acc, confusion = train_fold(train_rows, test_rows, labels, args.data,
                                    args.epochs, device)
        accs.append(acc)
        total_confusion += confusion
        _, _, f1 = per_class_f1(confusion)
        print(f"fold {i + 1}/{args.folds}  test {len(test_rows):4d}  "
              f"acc {acc:.3f}  macro F1 {f1.mean():.3f}")

    precision, recall, f1 = per_class_f1(total_confusion)
    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))

    print(f"\n{'=' * 52}")
    print(f"mode       {'grouped by recording run' if grouped else 'UNGROUPED (leaky)'}")
    print(f"accuracy   {mean_acc:.3f} +/- {std_acc:.3f}  "
          f"(range {min(accs):.3f} to {max(accs):.3f})")
    print(f"macro F1   {f1.mean():.3f}   pooled over "
          f"{int(total_confusion.sum())} clips, every video tested once")
    print(f"per class  {int(total_confusion.sum() / len(labels))} clips each "
          "on average")

    weakest = np.argsort(f1)[:5]
    print("weakest:   " + ", ".join(f"{labels[i]} ({f1[i]:.2f})"
                                    for i in weakest))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "folds": args.folds,
        "group_by": args.group_by,
        "leakage_controlled": grouped,
        "recording_runs": total_runs,
        "runs_straddling_folds": straddling,
        "fold_accuracies": accs,
        "test_acc": mean_acc,
        "acc_std": std_acc,
        "macro_f1": float(f1.mean()),
        "labels": labels,
        "per_class": {labels[i]: {"precision": float(precision[i]),
                                  "recall": float(recall[i]),
                                  "f1": float(f1[i])}
                      for i in range(len(labels))},
        "confusion": total_confusion.tolist(),
    }, indent=2))
    print(f"\nwrote {out}")
    print("make_figures.py works on this file too:")
    print(f"  python src/make_figures.py --metrics {out} --out figures/cv")
    if grouped:
        print("\nFor the comparison number, run the leaky version too:")
        print("  python src/cross_validate.py --folds 5 --group-by none "
              "--out checkpoints/cv_metrics_ungrouped.json")
        print("Report both. The gap is the size of the leakage.")


if __name__ == "__main__":
    main()
