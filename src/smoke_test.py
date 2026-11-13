"""End-to-end check of every stage, headless, on dataset videos.

    python src/smoke_test.py                 # 6 videos from the held-out split
    python src/smoke_test.py --n 12
    python src/smoke_test.py --video "data/INCLUDE/Places/19. House/MVI_3268.MOV"

Runs each stage in order and reports PASS or FAIL for each, so a failure names
the stage instead of surfacing as a blank Streamlit panel.

    0  environment      versions, model bundles, checkpoint, cached data
    1  extraction       video -> 282-float feature vectors
    2  buffer           vectors -> the exact shape the model was trained on
    3  recognition      sequence -> sign label, checked against ground truth
    4  phrase           prediction stream -> a finished phrase
    5  language         phrase -> English sentence
    6  speech           text to speech available

Test videos are chosen with the same seed and splitter as training, so the
recognition numbers here are on clips the model never saw. Passing `--video`
overrides that, and a training clip will score far too well.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results = []


def stage(n, name, fn):
    """Run one stage, catch anything, record the outcome."""
    print(f"\n[{n}] {name}")
    print("-" * 56)
    t0 = time.perf_counter()
    try:
        status, detail = fn()
    except Exception as e:
        status, detail = FAIL, f"{type(e).__name__}: {e}"
        import traceback
        traceback.print_exc(limit=3)
    ms = (time.perf_counter() - t0) * 1000
    results.append((n, name, status, detail, ms))
    print(f"  {status}  {detail}  ({ms:.0f} ms)")
    return status == PASS


# ---- stage 0 --------------------------------------------------------------

def check_environment():
    import cv2
    from config import CKPT_DIR, DATA_DIR, HAND_TASK, POSE_TASK

    notes = []
    missing = []

    try:
        import mediapipe as mp
        notes.append(f"mediapipe {mp.__version__}")
    except ImportError:
        missing.append("mediapipe")

    notes.append(f"opencv {cv2.__version__}")
    if cv2.__version__.startswith("5."):
        missing.append("opencv 5.x returns black frames on macOS; pin below 5")

    try:
        import torch
        dev = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")
        notes.append(f"torch {torch.__version__} on {dev}")
    except ImportError:
        missing.append("torch")

    for p, label in [(HAND_TASK, "hand_landmarker.task"),
                     (POSE_TASK, "pose_landmarker_lite.task")]:
        if not Path(p).exists():
            missing.append(f"{label} (run scripts/download_models.sh)")

    ckpt = CKPT_DIR / "best.pt"
    if not ckpt.exists():
        missing.append("checkpoints/best.pt (train first)")

    index = Path(DATA_DIR) / "index.csv"
    if not index.exists():
        missing.append("data/index.csv (extract first)")

    print("  " + " | ".join(notes))
    if missing:
        return FAIL, "missing: " + "; ".join(missing)
    return PASS, "all dependencies, models and data present"


# ---- helpers --------------------------------------------------------------

def held_out_videos(n):
    """Videos from the test split, chosen with the training seed and splitter."""
    from config import SEED
    from dataset import load_index, stratified_split

    rows = load_index("data")
    _, _, test_rows = stratified_split(rows, seed=SEED)
    rng = np.random.default_rng(0)
    picks = [test_rows[i] for i in
             rng.choice(len(test_rows), size=min(n, len(test_rows)),
                        replace=False)]
    return [(Path(r["src"]), r["label"]) for r in picks]


# ---- stages 1 to 3 --------------------------------------------------------

def run_recognition(videos):
    """Extraction, buffering and recognition, measured over several clips."""
    from extractor import LandmarkExtractor
    from realtime import RollingBuffer, SignRecognizer
    from config import FEATURE_DIM, SEQ_LEN

    state = {}

    def extraction():
        missing = [v for v, _ in videos if not v.exists()]
        if missing:
            return FAIL, f"{len(missing)} video(s) not found, first: {missing[0]}"
        seqs = []
        with LandmarkExtractor(running_mode="video") as ext:
            for path, label in videos:
                seq = ext.video(path)
                seqs.append((seq, label, path))
        state["seqs"] = seqs
        widths = {s.shape[1] for s, _, _ in seqs}
        if widths != {FEATURE_DIM}:
            return FAIL, f"feature width {widths}, expected {FEATURE_DIM}"
        hand_rate = np.mean([float((s[:, -2:] > 0).any(axis=1).mean())
                             for s, _, _ in seqs])
        return PASS, (f"{len(seqs)} clips, width {FEATURE_DIM}, "
                      f"hands detected in {hand_rate:.0%} of frames")

    def buffering():
        if "seqs" not in state:
            return SKIP, "extraction failed"
        built = []
        for seq, label, path in state["seqs"]:
            buf = RollingBuffer()
            for vec in seq:
                buf.push(vec)
            if not buf.ready:
                return FAIL, f"buffer never filled for {path.name}"
            built.append((buf.sequence(with_motion=True), label, path))
        state["built"] = built
        shapes = {b.shape for b, _, _ in built}
        want = (SEQ_LEN, FEATURE_DIM * 2)
        if shapes != {want}:
            return FAIL, f"got {shapes}, model expects {want}"
        return PASS, f"every clip -> {want}, matching training"

    def recognition():
        if "built" not in state:
            return SKIP, "buffering failed"
        recog = SignRecognizer()
        state["recog"] = recog
        correct, rows = 0, []
        for seq, truth, path in state["built"]:
            if not recog.use_motion:
                seq = seq[:, :seq.shape[1] // 2]
            pred, conf = recog.predict(seq)
            ok = pred == truth
            correct += ok
            rows.append((truth, pred, conf, ok))
            print(f"    {'OK ' if ok else '   '} {truth:18} -> "
                  f"{pred:18} {conf:.2f}")
        state["preds"] = rows
        n = len(rows)
        if correct == 0:
            return FAIL, (f"0/{n} correct. the model runs but predicts nothing "
                          "right, which usually means the inference features "
                          "are built differently from the training features")
        return PASS, f"{correct}/{n} correct on held-out clips"

    return extraction, buffering, recognition, state


# ---- stage 4 --------------------------------------------------------------

def check_phrase(state):
    """Feed the detector a realistic prediction stream and expect one phrase."""
    from phrase import PhraseConfig, PhraseDetector

    if "preds" not in state:
        return SKIP, "recognition failed"

    cfg = PhraseConfig(stable_frames=5, hands_down_frames=12)
    det = PhraseDetector(cfg)
    t = 0.0

    # three signs held for 25 frames each, hands down between them, then a
    # final hands-down run to close the phrase
    for sign in [p for _, p, _, _ in state["preds"]][:3]:
        for _ in range(25):
            det.update(sign, 0.95, True, now=t)
            t += 0.04
        for _ in range(3):
            det.update(None, 0.0, False, now=t)
            t += 0.04

    phrase = None
    for _ in range(cfg.hands_down_frames + 2):
        out = det.update(None, 0.0, False, now=t)
        t += 0.04
        if out:
            phrase = out
            break

    if not phrase:
        return FAIL, "hands-down did not emit a phrase"
    expected = [p for _, p, _, _ in state["preds"]][:3]
    state["phrase"] = phrase
    if phrase != expected:
        return FAIL, f"emitted {phrase}, expected {expected}"
    return PASS, f"3 gestures -> {len(phrase)} words: {' '.join(phrase).upper()}"


# ---- stage 5 --------------------------------------------------------------

def check_language(state, temperature):
    from llm_eval import faithfulness
    from storyteller import Storyteller

    glosses = state.get("phrase") or ["i", "happy"]
    teller = Storyteller(temperature=temperature)

    if teller.provider == "none":
        sentence = teller.add(glosses)
        print(f"    template: {sentence}")
        return FAIL, ("no API key, so this is the offline template, not the "
                      "model. put a key in .env")

    sentence = teller.add(glosses)
    print(f"    {' '.join(glosses).upper()}  ->  {sentence}")
    if teller.last_error:
        return FAIL, f"API call failed: {teller.last_error}"

    f = faithfulness(glosses, sentence)
    detail = f"{teller.provider}/{teller.model or 'default'}: \"{sentence}\""
    if f["invented_entities"]:
        detail += f"  (invented {f['invented_entities']})"
    if f["dropped"]:
        detail += f"  (dropped {f['dropped']})"
    state["sentence"] = sentence
    return PASS, detail


# ---- stage 6 --------------------------------------------------------------

def check_speech(state):
    from tts import available
    if not available():
        return SKIP, "macOS `say` not found; speech is optional"
    return PASS, "macOS `say` available"


# ---- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6,
                    help="held-out clips to test recognition on")
    ap.add_argument("--video", action="append",
                    help="use these clips instead (may be a training clip, "
                         "which will score misleadingly well)")
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    print("=" * 56)
    print("SIGN STORYTELLER: END-TO-END CHECK")
    print("=" * 56)

    if not stage(0, "environment", check_environment):
        summarise()
        raise SystemExit(1)

    if args.video:
        videos = [(Path(v), Path(v).parent.name.split(". ")[-1]
                   .lower().replace(" ", "_")) for v in args.video]
        print(f"\nusing {len(videos)} clip(s) given on the command line")
    else:
        videos = held_out_videos(args.n)
        print(f"\n{len(videos)} clips from the held-out test split")

    extraction, buffering, recognition, state = run_recognition(videos)
    stage(1, "extraction: video -> features", extraction)
    stage(2, "buffer: features -> model input", buffering)
    stage(3, "recognition: sequence -> sign", recognition)
    stage(4, "phrase: predictions -> phrase", lambda: check_phrase(state))
    stage(5, "language: phrase -> sentence",
          lambda: check_language(state, args.temperature))
    stage(6, "speech", lambda: check_speech(state))

    summarise()


def summarise():
    print("\n" + "=" * 56)
    print("SUMMARY")
    print("=" * 56)
    for n, name, status, detail, ms in results:
        print(f"  [{n}] {status:4}  {name}")
        if status != PASS:
            print(f"         {detail}")
    failed = [r for r in results if r[2] == FAIL]
    skipped = [r for r in results if r[2] == SKIP]
    print()
    if failed:
        print(f"{len(failed)} stage(s) failed. First: "
              f"[{failed[0][0]}] {failed[0][1]}")
        print(f"  {failed[0][3]}")
        raise SystemExit(1)
    print(f"all stages passed"
          + (f", {len(skipped)} skipped" if skipped else ""))


if __name__ == "__main__":
    main()
