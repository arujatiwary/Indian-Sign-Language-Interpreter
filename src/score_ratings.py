"""Turn filled rating sheets into numbers for the report.

    python src/score_ratings.py checkpoints/rating_sheet_asha.md \
                                checkpoints/rating_sheet_ravi.md

One rater gives an opinion. Two give an opinion plus a measure of how much that
opinion depends on who is holding the pen, which is the part worth reporting.

Reads the sheets written by llm_eval.rating_sheet, so the table under each item
is `| meaning | grammar | context | no_extra |` with one row of scores.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from llm_eval import RATING_CRITERIA  # noqa: E402

KEYS = [k for k, _ in RATING_CRITERIA]


def parse(path):
    """Return an (items x criteria) array of scores, NaN where left blank."""
    text = Path(path).read_text()
    rows = []
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        cells = None
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("|") or set(line) <= set("|- "):
                continue
            if any(k in line for k in KEYS):
                continue                      # the header row
            cells = [c.strip() for c in line.strip("|").split("|")]
            break
        if cells is None:
            continue
        rows.append([float(c) if re.fullmatch(r"[1-5]", c) else np.nan
                     for c in cells[:len(KEYS)]])
    if not rows:
        raise SystemExit(f"no score rows found in {path}")
    return np.array(rows, dtype=float)


def agreement(a, b):
    """Exact agreement and within-one agreement across every filled cell."""
    both = ~np.isnan(a) & ~np.isnan(b)
    if not both.any():
        return None
    d = np.abs(a[both] - b[both])
    return {"cells": int(both.sum()),
            "exact": float((d == 0).mean()),
            "within_one": float((d <= 1).mean()),
            "mean_abs_diff": float(d.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheets", nargs="+")
    args = ap.parse_args()

    scored = {}
    for path in args.sheets:
        arr = parse(path)
        scored[Path(path).stem] = arr
        filled = int((~np.isnan(arr)).sum())
        total = arr.size
        print(f"{Path(path).stem:28} {len(arr):3d} items, "
              f"{filled}/{total} cells filled")
        if filled < total:
            print(f"  {total - filled} blank; blanks are ignored, not scored 0")

    # Strip whatever the sheets have in common so two columns are not both
    # printed as "rating_sheet".
    names = list(scored)
    short = names
    if len(names) > 1:
        pre = 0
        while all(len(n) > pre + 1 and n[pre] == names[0][pre] for n in names):
            pre += 1
        short = [n[pre:].lstrip("_- ") or n for n in names]

    print()
    print(f"  {'criterion':12}" + "".join(f"{n[:12]:>14}" for n in short)
          + f"{'pooled':>10}")
    stack = np.stack(list(scored.values()))
    for j, key in enumerate(KEYS):
        per = [f"{np.nanmean(a[:, j]):14.2f}" for a in scored.values()]
        print(f"  {key:12}" + "".join(per)
              + f"{np.nanmean(stack[:, :, j]):10.2f}")
    per = [f"{np.nanmean(a):14.2f}" for a in scored.values()]
    print(f"  {'overall':12}" + "".join(per) + f"{np.nanmean(stack):10.2f}")

    if len(names) < 2:
        print("\none rater only. Add a second sheet to report agreement; a "
              "single rater's mean is not a measurement.")
        return
    print("\n  inter-rater agreement")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = scored[names[i]], scored[names[j]]
            if a.shape != b.shape:
                print(f"  {names[i]} vs {names[j]}: sheets differ in length "
                      f"({a.shape[0]} vs {b.shape[0]} items), skipped")
                continue
            g = agreement(a, b)
            print(f"  {names[i]} vs {names[j]}: exact {g['exact']:.0%}, "
                  f"within one point {g['within_one']:.0%}, "
                  f"mean gap {g['mean_abs_diff']:.2f} over {g['cells']} cells")


if __name__ == "__main__":
    main()
