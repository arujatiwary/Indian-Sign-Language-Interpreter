"""Train the sign classifier on cached landmark sequences.

    python src/train.py --data data --epochs 60
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from config import BATCH_SIZE, CKPT_DIR, EPOCHS, FEATURE_DIM, LR, SEED  # noqa: E402
from dataset import SignSequences, load_index, stratified_split  # noqa: E402
from model import SignLSTM  # noqa: E402


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")   # Apple Silicon
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    loss_sum, n = 0.0, 0
    lossf = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += lossf(logits, y).item() * y.size(0)
        n += y.size(0)
        for t, p in zip(y.cpu().numpy(), logits.argmax(1).cpu().numpy()):
            confusion[t, p] += 1
    acc = confusion.trace() / max(1, confusion.sum())
    return loss_sum / max(1, n), acc, confusion


def per_class_f1(confusion):
    tp = np.diag(confusion).astype(float)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--no-motion", action="store_true")
    ap.add_argument("--vocab", help="text file, one sign per line; trains on "
                                    "only those signs")
    ap.add_argument("--out", default="best.pt",
                    help="checkpoint filename inside checkpoints/")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    rows = load_index(args.data)

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
                + "\ncheck spelling against data/labels.txt"
            )
        rows = [r for r in rows if r["label"] in wanted]
        print(f"vocab filter: {len(rows)} videos across {len(wanted)} signs")

    labels = sorted({r["label"] for r in rows})
    train_rows, val_rows, test_rows = stratified_split(rows, seed=SEED)
    print(f"{len(labels)} signs | train {len(train_rows)} val {len(val_rows)} test {len(test_rows)}")

    use_motion = not args.no_motion
    input_dim = FEATURE_DIM * (2 if use_motion else 1)

    def make(rows_, train):
        ds = SignSequences(rows_, args.data, labels, train=train, use_motion=use_motion)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train, num_workers=0)

    train_loader, val_loader, test_loader = make(train_rows, True), make(val_rows, False), make(test_rows, False)

    # feature statistics from the training split only
    stack = np.concatenate([train_loader.dataset[i][0].numpy() for i in range(len(train_rows))])
    feat_mean, feat_std = stack.mean(axis=0), stack.std(axis=0)

    device = pick_device()
    model = SignLSTM(input_dim, len(labels)).to(device)
    model.set_stats(feat_mean, feat_std)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_acc, history = 0.0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * y.size(0)
            n += y.size(0)
        sched.step()

        val_loss, val_acc, _ = evaluate(model, val_loader, device, len(labels))
        history.append({"epoch": epoch, "train_loss": total / max(1, n),
                        "val_loss": val_loss, "val_acc": val_acc})
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | train {total/max(1,n):.3f} | "
                  f"val {val_loss:.3f} | val acc {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "labels": labels,
                        "input_dim": input_dim, "use_motion": use_motion},
                       CKPT_DIR / args.out)

    ckpt = torch.load(CKPT_DIR / args.out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_loss, test_acc, confusion = evaluate(model, test_loader, device, len(labels))
    precision, recall, f1 = per_class_f1(confusion)

    print(f"\ntest acc {test_acc:.3f} | macro f1 {f1.mean():.3f}")
    worst = np.argsort(f1)[:5]
    print("weakest signs:", ", ".join(f"{labels[i]} ({f1[i]:.2f})" for i in worst))

    metrics_name = Path(args.out).stem + "_metrics.json"
    (CKPT_DIR / metrics_name).write_text(json.dumps({
        "test_acc": test_acc, "test_loss": test_loss, "macro_f1": float(f1.mean()),
        "labels": labels,
        "per_class": {labels[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1[i])} for i in range(len(labels))},
        "confusion": confusion.tolist(), "history": history,
    }, indent=2))


if __name__ == "__main__":
    main()
