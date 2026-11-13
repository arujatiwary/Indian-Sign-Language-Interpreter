"""Tests for leakage-controlled cross-validation folds.

These load cross_validate.py's splitter functions without importing torch, so
they run anywhere. The property under test is the one that decides whether the
reported accuracy means anything: a recording run must never straddle a fold.
"""

import collections
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load():
    """Exec just the splitter section, skipping the torch imports."""
    src = (ROOT / "src" / "cross_validate.py").read_text()
    section = src[src.index("def recording_runs"):src.index("def train_fold")]
    ns = {"re": re, "defaultdict": collections.defaultdict, "np": np, "SEED": 42}
    exec(section, ns)
    return ns


NS = _load()


def make_rows(n_labels=6, runs_per_label=5, takes=3):
    """Synthetic index rows with a known run structure."""
    rows, mvi = [], 1000
    for li in range(n_labels):
        for _ in range(runs_per_label):
            for t in range(takes):
                rows.append({"file": f"{mvi}.npy", "label": f"sign{li}",
                             "src": f"data/X/{li}. sign{li}/MVI_{mvi}.MOV"})
                mvi += 1
            mvi += 50          # gap marks a new sitting
    return rows


def test_consecutive_numbers_form_one_run():
    rows = make_rows(n_labels=1, runs_per_label=4, takes=3)
    runs = NS["recording_runs"](rows)
    assert len(set(runs.values())) == 4


def test_a_gap_starts_a_new_run():
    rows = [
        {"file": "a.npy", "label": "x", "src": "d/1. x/MVI_100.MOV"},
        {"file": "b.npy", "label": "x", "src": "d/1. x/MVI_101.MOV"},
        {"file": "c.npy", "label": "x", "src": "d/1. x/MVI_900.MOV"},
    ]
    runs = NS["recording_runs"](rows)
    assert runs["a.npy"] == runs["b.npy"]
    assert runs["c.npy"] != runs["a.npy"]


def test_runs_never_merge_across_labels():
    """Two signs recorded back to back must not share a run."""
    rows = [
        {"file": "a.npy", "label": "x", "src": "d/1. x/MVI_100.MOV"},
        {"file": "b.npy", "label": "y", "src": "d/2. y/MVI_101.MOV"},
    ]
    runs = NS["recording_runs"](rows)
    assert runs["a.npy"] != runs["b.npy"]


def test_unparsable_filenames_become_their_own_run():
    """Conservative: never merge clips we cannot order."""
    rows = [
        {"file": "a.npy", "label": "x", "src": "d/1. x/clip_one.MOV"},
        {"file": "b.npy", "label": "x", "src": "d/1. x/clip_two.MOV"},
    ]
    runs = NS["recording_runs"](rows)
    assert runs["a.npy"] != runs["b.npy"]


@pytest.mark.parametrize("k", [3, 4, 5])
def test_no_run_straddles_a_fold(k):
    """The property the whole file exists for."""
    rows = make_rows()
    runs = NS["recording_runs"](rows)
    folds = NS["grouped_folds"](rows, k)
    seen = [{runs[r["file"]] for r in f} for f in folds]
    for i in range(k):
        for j in range(i + 1, k):
            assert not (seen[i] & seen[j]), f"run shared by folds {i} and {j}"


@pytest.mark.parametrize("k", [3, 4, 5])
def test_every_clip_used_exactly_once(k):
    rows = make_rows()
    folds = NS["grouped_folds"](rows, k)
    files = [r["file"] for f in folds for r in f]
    assert len(files) == len(rows)
    assert len(set(files)) == len(rows)


def test_folds_are_balanced():
    rows = make_rows()
    sizes = [len(f) for f in NS["grouped_folds"](rows, 5)]
    assert max(sizes) - min(sizes) <= max(3, 0.15 * np.mean(sizes))


def test_every_class_present_when_runs_allow_it():
    rows = make_rows(n_labels=6, runs_per_label=5)
    folds = NS["grouped_folds"](rows, 5)
    for i, f in enumerate(folds):
        present = {r["label"] for r in f}
        assert len(present) == 6, f"fold {i} missing classes"


def test_max_usable_folds_reports_the_rarest_class():
    rows = make_rows(n_labels=3, runs_per_label=5)
    rows += [{"file": "r1.npy", "label": "rare", "src": "d/9. rare/MVI_8000.MOV"},
             {"file": "r2.npy", "label": "rare", "src": "d/9. rare/MVI_8500.MOV"}]
    assert NS["max_usable_folds"](rows) == 2


def test_ungrouped_split_leaks_and_grouped_does_not():
    """The comparison that justifies the default."""
    rows = make_rows()
    runs = NS["recording_runs"](rows)

    def leaked(folds):
        n = 0
        for i, test in enumerate(folds):
            train = {runs[r["file"]] for j, f in enumerate(folds) if j != i
                     for r in f}
            n += sum(1 for r in test if runs[r["file"]] in train)
        return n

    assert leaked(NS["grouped_folds"](rows, 5)) == 0
    assert leaked(NS["stratified_folds"](rows, 5)) > 0
