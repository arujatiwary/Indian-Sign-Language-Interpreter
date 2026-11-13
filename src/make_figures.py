"""Build result figures from a finished training run.

    python src/make_figures.py --metrics checkpoints/metrics.json --out figures

Produces:
    training_curves.png   loss and validation accuracy against epoch
    confusion_matrix.png  full confusion matrix
    per_class_f1.png      F1 per sign, sorted, with the class size on each bar
    confusions.png        the sign pairs that actually get mixed up

Everything is drawn from metrics.json, so the numbers on the slide are the
numbers the run produced.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#1a1a1a"
ACCENT = "#2b6cb0"
WARM = "#c05621"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)


def training_curves(m, out):
    h = m.get("history", [])
    if not h:
        print("  no history in metrics, skipping training curves")
        return
    ep = [r["epoch"] for r in h]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

    a1.plot(ep, [r["train_loss"] for r in h], color=ACCENT, label="train")
    a1.plot(ep, [r["val_loss"] for r in h], color=WARM, label="validation")
    a1.set_xlabel("epoch")
    a1.set_ylabel("cross-entropy loss")
    a1.set_title("Loss")
    a1.legend(frameon=False, fontsize=9)
    _style(a1)

    acc = [r["val_acc"] for r in h]
    a2.plot(ep, acc, color=ACCENT)
    best = int(np.argmax(acc))
    a2.scatter([ep[best]], [acc[best]], color=WARM, zorder=5, s=40)
    a2.annotate(f"{acc[best]:.3f} at epoch {ep[best]}",
                (ep[best], acc[best]), textcoords="offset points",
                xytext=(-10, -18), fontsize=9, color=WARM, ha="right")
    a2.set_xlabel("epoch")
    a2.set_ylabel("validation accuracy")
    a2.set_ylim(0, 1)
    a2.set_title("Validation accuracy")
    _style(a2)

    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  training_curves.png")


def confusion_matrix(m, out):
    labels = m["labels"]
    C = np.array(m["confusion"])
    n = len(labels)
    size = max(7, min(16, n * 0.28))

    fig, ax = plt.subplots(figsize=(size, size * 0.92))
    ax.imshow(C, cmap="Blues", interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    fs = 8 if n <= 25 else 6
    ax.set_xticklabels(labels, rotation=90, fontsize=fs)
    ax.set_yticklabels(labels, fontsize=fs)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix, {n} signs "
                 f"(test accuracy {m['test_acc']:.3f})", pad=12)

    # mark the off-diagonal cells so errors are visible at a glance
    for i in range(n):
        for j in range(n):
            if i != j and C[i, j] > 0:
                ax.text(j, i, str(C[i, j]), ha="center", va="center",
                        fontsize=fs, color=WARM, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  confusion_matrix.png")


def per_class_f1(m, out):
    per = m["per_class"]
    C = np.array(m["confusion"])
    labels = m["labels"]
    support = {labels[i]: int(C[i].sum()) for i in range(len(labels))}

    items = sorted(per.items(), key=lambda kv: kv[1]["f1"])
    names = [k for k, _ in items]
    f1 = [v["f1"] for _, v in items]
    macro = float(np.mean(f1))

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.26)))
    colors = [WARM if v < macro else ACCENT for v in f1]
    ax.barh(range(len(names)), f1, color=colors, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1")
    ax.axvline(macro, color=INK, linestyle="--", linewidth=1)
    ax.text(macro + 0.01, len(names) - 0.5, f"macro {macro:.3f}",
            fontsize=9, color=INK, va="top")

    for i, name in enumerate(names):
        ax.text(1.02, i, f"n={support.get(name, 0)}", fontsize=7,
                color="#666666", va="center")

    ax.set_title("F1 per sign, with test set size")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "per_class_f1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  per_class_f1.png")


def confusions(m, out, top=12):
    """The pairs actually confused. Far more readable than a sparse matrix."""
    labels = m["labels"]
    C = np.array(m["confusion"])
    pairs = [(labels[i], labels[j], int(C[i, j]))
             for i in range(len(labels)) for j in range(len(labels))
             if i != j and C[i, j] > 0]
    if not pairs:
        print("  no off-diagonal errors, skipping confusions plot")
        return
    pairs.sort(key=lambda p: -p[2])
    pairs = pairs[:top]

    names = [f"{t}  \u2192  {p}" for t, p, _ in pairs][::-1]
    counts = [c for _, _, c in pairs][::-1]

    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.34)))
    ax.barh(range(len(names)), counts, color=WARM, height=0.65)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("test clips misclassified")
    ax.set_title("Where the model confuses signs (true \u2192 predicted)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "confusions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  confusions.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="checkpoints/metrics.json")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    path = Path(args.metrics)
    if not path.exists():
        raise SystemExit(f"{path} not found. train first, or pass --metrics.")
    m = json.loads(path.read_text())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"writing to {out}/")

    training_curves(m, out)
    confusion_matrix(m, out)
    per_class_f1(m, out)
    confusions(m, out)

    print(f"\ntest accuracy {m['test_acc']:.3f}")
    print(f"macro F1      {np.mean([v['f1'] for v in m['per_class'].values()]):.3f}")
    print(f"test clips    {int(np.array(m['confusion']).sum())}")


if __name__ == "__main__":
    main()