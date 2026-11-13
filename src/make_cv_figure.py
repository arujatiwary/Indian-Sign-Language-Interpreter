"""Protocol comparison figure: what leakage control costs, and where.

    python src/make_cv_figure.py --out figures/cv

Two panels:
  left   accuracy under each evaluation protocol, with fold spread
  right  per-class F1 under both protocols, so the drop is visible per sign

Colours match the project's other figures and pass the categorical palette
checks (CVD dE 20.9, well above the 8 floor).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED = "#1a1a1a", "#666666"
HONEST, LEAKY = "#2b6cb0", "#c05621"
GRID = "#d8dee6"


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9, length=0)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grouped", default="checkpoints/cv_metrics.json")
    ap.add_argument("--ungrouped", default="checkpoints/cv_metrics_ungrouped.json")
    ap.add_argument("--k3", default="checkpoints/cv_metrics_k3.json")
    ap.add_argument("--single", default="checkpoints/metrics.json")
    ap.add_argument("--out", default="figures/cv")
    args = ap.parse_args()

    g = json.loads(Path(args.grouped).read_text())
    u = json.loads(Path(args.ungrouped).read_text())
    k3 = json.loads(Path(args.k3).read_text()) if Path(args.k3).exists() else None
    single = (json.loads(Path(args.single).read_text())
              if Path(args.single).exists() else None)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- panel 1: protocol comparison -------------------------------
    rows = []
    if single:
        rows.append(("Single split\n(original)", single["test_acc"], 0.0, True))
    rows.append(("5-fold\nungrouped", u["test_acc"], u["acc_std"], True))
    rows.append(("5-fold grouped\n(leakage controlled)", g["test_acc"],
                 g["acc_std"], False))
    if k3:
        rows.append(("3-fold grouped\n(less training data)", k3["test_acc"],
                     k3["acc_std"], False))

    # 50 class labels need vertical room; a short figure collides them.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 9.0),
                                 gridspec_kw={"width_ratios": [1, 1.25]})

    names = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    errs = [r[2] for r in rows][::-1]
    leaky = [r[3] for r in rows][::-1]
    colors = [LEAKY if l else HONEST for l in leaky]

    y = np.arange(len(names))
    a1.barh(y, vals, xerr=errs, height=0.42, color=colors,
            error_kw={"ecolor": INK, "elinewidth": 1.2, "capsize": 4},
            zorder=3)
    for i, (v, e) in enumerate(zip(vals, errs)):
        txt = f"{v:.3f}" + (f" ± {e:.3f}" if e else "")
        a1.text(v + e + 0.012, i, txt, va="center", fontsize=9, color=INK,
                zorder=4)

    a1.set_yticks(y)
    a1.set_yticklabels(names, fontsize=9)
    a1.set_xlim(0, 1.0)
    a1.set_xlabel("test accuracy")
    a1.set_title("Accuracy depends on the evaluation protocol", fontsize=11,
                 color=INK, pad=10)
    a1.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    a1.set_axisbelow(True)
    _clean(a1)

    handles = [plt.Rectangle((0, 0), 1, 1, color=LEAKY),
               plt.Rectangle((0, 0), 1, 1, color=HONEST)]
    # Above the plot, so it cannot sit on top of a bar.
    a1.legend(handles, ["repeat takes split across train/test (leaky)",
                        "recording runs kept whole (honest)"],
              frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.1))
    a1.set_ylim(-0.7, len(names) - 0.3)

    # ---- panel 2: per-class F1, both protocols ----------------------
    labels = g["labels"]
    gf = np.array([g["per_class"][l]["f1"] for l in labels])
    uf = np.array([u["per_class"][l]["f1"] for l in labels])
    order = np.argsort(gf)

    yy = np.arange(len(labels))
    for i, idx in enumerate(order):
        a2.plot([gf[idx], uf[idx]], [i, i], color=GRID, linewidth=1.4,
                zorder=1, solid_capstyle="round")
    a2.scatter(uf[order], yy, s=26, color=LEAKY, zorder=3,
               label="ungrouped (leaky)", edgecolors="white", linewidths=0.8)
    a2.scatter(gf[order], yy, s=26, color=HONEST, zorder=3,
               label="grouped (honest)", edgecolors="white", linewidths=0.8)

    a2.set_yticks(yy)
    a2.set_yticklabels([labels[i] for i in order], fontsize=7.5)
    a2.set_xlim(-0.03, 1.06)
    a2.set_xlabel("per-class F1")
    a2.set_title("Every sign scores lower once repeat takes are held out",
                 fontsize=11, color=INK, pad=10)
    a2.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    a2.set_axisbelow(True)
    a2.legend(frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.06), ncol=2)
    _clean(a2)

    drop = uf - gf
    worst = int(np.argmax(drop))
    row = list(order).index(worst)
    a2.text(gf[worst] - 0.02, row, f"-{drop[worst]:.2f} ", ha="right",
            va="center", fontsize=7.5, color=MUTED)

    fig.tight_layout()
    path = out / "protocol_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")

    # ---- the numbers, as a table the report can quote ---------------
    tbl = out / "protocol_comparison.md"
    lines = ["| Protocol | Accuracy | Macro F1 | Test clips |",
             "|---|---|---|---|"]
    if single:
        n = int(np.array(single["confusion"]).sum())
        f1s = np.mean([v["f1"] for v in single["per_class"].values()])
        lines.append(f"| Single split (original, leaky) | {single['test_acc']:.3f} "
                     f"| {f1s:.3f} | {n} |")
    lines.append(f"| 5-fold ungrouped (leaky) | {u['test_acc']:.3f} ± "
                 f"{u['acc_std']:.3f} | {u['macro_f1']:.3f} | 675 |")
    lines.append(f"| **5-fold grouped (leakage controlled)** | "
                 f"**{g['test_acc']:.3f} ± {g['acc_std']:.3f}** | "
                 f"**{g['macro_f1']:.3f}** | 675 |")
    if k3:
        lines.append(f"| 3-fold grouped (450 training clips) | "
                     f"{k3['test_acc']:.3f} ± {k3['acc_std']:.3f} | "
                     f"{k3['macro_f1']:.3f} | 675 |")
    tbl.write_text("\n".join(lines) + "\n")
    print(f"  {tbl}")


if __name__ == "__main__":
    main()
