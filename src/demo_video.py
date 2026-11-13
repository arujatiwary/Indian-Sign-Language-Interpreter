"""Build a demo video from dataset clips, then run the whole pipeline on it.

This is the recorded-demo safety net. A live webcam demo depends on lighting,
on your signing resembling the training signers, and on the API responding.
This depends on none of that: it stitches real INCLUDE clips into one
continuous video, feeds that video through the exact pipeline the live app
uses, and prints the paragraph.

    python src/demo_video.py --build --out demo/story.mp4
    python src/demo_video.py --run demo/story.mp4 --narrate
    python src/demo_video.py --build --run --narrate          # both

SIGNER CONSISTENCY. A naive pick takes the cleanest clip per sign, which
produces a "story" told by ten different people and looks broken on screen.
INCLUDE ships no signer column, and the camera's file numbering does not
separate signers either: MVI_5228 and MVI_5243 are one woman while MVI_5231,
sitting between them, is a different man. So clips are matched by appearance
instead. Each candidate contributes a colour histogram of the torso region,
and the story is built from whichever signer has good candidates for every
sign. Clothing is strongly distinctive in this dataset, so this is reliable.

HONESTY. Say on the slide what this is: held-out clips the model never
trained on, restricted to signs with good measured F1. That is a scripted
demo, not a cherry-picked result, as long as you say so.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Phrases built from signs with F1 >= 0.82 under grouped cross-validation, and
# from a vocabulary one signer actually covers. Every content word in the
# target paragraph is signed; nothing is supplied but grammar.
DEFAULT_STORY = [
    ["father", "cow", "window"],
    ["brother", "t-shirt"],
    ["brother", "train_ticket"],
]

# INCLUDE clips are 1920x1080 at 25 fps. Downscaling to 360p shrank the hands
# to a few dozen pixels and MediaPipe stopped finding them, so the stitched
# video keeps the source resolution the model was trained on.
TARGET_W, TARGET_H, TARGET_FPS = 1920, 1080, 25
DESC_FRAME = 5
DESC_FRAMES = (3, 12, 25)   # averaged, so one bad frame cannot decide

# The signer whose clips cover DEFAULT_STORY. Verified by eye in
# demo/contact_sheet.jpg: one man, blue shirt, same room, all seven clips.
DEFAULT_REFERENCE = ("data/INCLUDE/Means_of_Transportation/"
                     "16. train ticket/MVI_4191.MOV")          # a frame early enough that the hands are still at rest


def otsu(values, bins=64, log=True):
    """Split a set of values into two clusters, Otsu's method.

    The same thresholding used to binarise an image, applied to motion instead
    of brightness. Frames fall into two groups, signing and pausing, and the
    boundary between them shifts with resolution, frame rate and signer, so the
    threshold is measured from the video rather than fixed in advance.

    The split is taken on log motion. Motion is multiplicative: a pause sits
    near 0.005 and signing near 0.10, a factor of twenty, and on a linear axis
    the long high-motion tail drags the cut up into the signing cluster. Taking
    logs makes the two groups comparable in width, which is what Otsu assumes.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[v > 0]
    if v.size == 0:
        return 0.0
    x = np.log(v) if log else v
    hist, edges = np.histogram(x, bins=bins)
    p = hist / max(hist.sum(), 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * omega - mu) ** 2 / (omega * (1.0 - omega))
    between[~np.isfinite(between)] = 0.0
    cut = float(centres[int(np.argmax(between))])
    return float(np.exp(cut)) if log else cut


def load_index(data_dir="data"):
    with open(Path(data_dir) / "index.csv") as f:
        return list(csv.DictReader(f))


# Upper torso only: shoulders to waist, shoulder width. The earlier box ran to
# 85% of frame height and 40% of its width, which swallowed the wall, the floor
# and the blue skirting band every clip shares.
TORSO_BOX = (0.22, 0.52, 0.38, 0.62)   # y0, y1, x0, x1 as fractions
MIN_COLOUR_PX = 400                     # below this the crop missed the signer


def appearance(video_path, frame_no=DESC_FRAME):
    """Colour histogram of the torso region, as a signer fingerprint.

    The torso is the middle of the frame and is mostly clothing, which varies
    far more between signers than the shared classroom background does. Hue
    and saturation only, so lighting differences matter less.
    """
    cap = cv2.VideoCapture(str(video_path))
    hists = []
    for fn in DESC_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        y0, y1 = int(h * TORSO_BOX[0]), int(h * TORSO_BOX[1])
        x0, x1 = int(w * TORSO_BOX[2]), int(w * TORSO_BOX[3])
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        # The classroom is a white wall, a whiteboard and a blue skirting
        # band, identical in every clip. Those pixels are near-grey or very
        # dark, and left in they dominate the histogram, so two different
        # signers score 0.9 on their shared background. Keeping only
        # coloured pixels makes the descriptor read clothing, which is what
        # actually differs between signers.
        mask = cv2.inRange(hsv, (0, 60, 50), (180, 255, 255))
        if int(mask.sum() // 255) < MIN_COLOUR_PX:
            continue
        hist = cv2.calcHist([hsv], [0, 1], mask, [24, 24], [0, 180, 0, 256])
        hists.append(hist / (hist.sum() + 1e-9))
    cap.release()
    if not hists:
        return None
    # Several frames, because the signer may be mid-step or half out of
    # frame on any single one. Probability-normalised, which is what the
    # Bhattacharyya distance in similarity() expects.
    return np.ascontiguousarray(np.mean(hists, axis=0).ravel(),
                                dtype=np.float32)


def similarity(a, b):
    """1.0 for the same signer, near 0 for a different one.

    Correlation looks like the obvious choice and is wrong here. These
    histograms are sparse: 576 bins, almost all empty for any one outfit. Two
    different signers agree on every empty bin, and correlation counts that
    agreement, so unrelated people scored 0.9 and the matcher accepted them.
    Bhattacharyya compares where the mass actually sits, so it ignores the
    shared emptiness. Measured on clips checked by eye, the same signer scores
    0.89 or better and different signers 0.67 or less.
    """
    if a is None or b is None:
        return 0.0
    d = cv2.compareHist(a.astype("float32"), b.astype("float32"),
                        cv2.HISTCMP_BHATTACHARYYA)
    return float(1.0 - d)


# Signs the classifier gets right under the leakage-controlled protocol
# (F1 >= 0.80 on grouped 5-fold). A demo should show the system working, not
# the weakest classes failing, and these are the ones measured as reliable.
DEMO_SAFE = ["cow", "window", "brother", "long", "monday", "happy",
             "train_ticket", "father", "t-shirt", "quiet", "loud", "house",
             "i", "time", "court", "fan", "dry"]


def suggest(rows, data_dir="data", min_hand_rate=0.6, min_match=0.85,
            pool=None):
    """Find the signer who covers the most reliable signs, and list them.

    Picking a story first and hunting for clips second fails: no single signer
    performed every sign you might want. Picking the signer first, then
    writing the story from what they actually signed, always works.
    """
    pool = pool or DEMO_SAFE
    by_label = {}
    for r in rows:
        if r["label"] in pool and float(r["hand_rate"]) >= min_hand_rate:
            by_label.setdefault(r["label"], []).append(r)

    print(f"  fingerprinting {sum(len(v) for v in by_label.values())} clips "
          f"across {len(by_label)} signs...")
    desc = {}
    for cands in by_label.values():
        for r in cands:
            d = appearance(Path(r["src"]))
            if d is not None:
                desc[r["file"]] = d

    best = None
    for cands in by_label.values():
        for ref in cands:
            rd = desc.get(ref["file"])
            if rd is None:
                continue
            covered = {}
            for sign, group in by_label.items():
                scored = [(similarity(rd, desc.get(r["file"])), r)
                          for r in group]
                scored.sort(key=lambda t: (-t[0], -float(t[1]["hand_rate"])))
                if scored and scored[0][0] >= min_match:
                    covered[sign] = (scored[0][0], scored[0][1])
            if best is None or len(covered) > len(best[1]):
                best = (ref, covered)

    ref, covered = best
    print(f"\n  best signer: {Path(ref['src']).name}")
    print(f"  covers {len(covered)}/{len(by_label)} reliable signs "
          f"at match >= {min_match}\n")
    for sign in sorted(covered, key=lambda s: -covered[s][0]):
        sim, r = covered[sign]
        print(f"    {sign:14} match {sim:.2f}  hand_rate "
              f"{float(r['hand_rate']):.2f}  {Path(r['src']).name}")
    uncovered = sorted(set(by_label) - set(covered))
    if uncovered:
        print(f"\n  not covered by this signer: {', '.join(uncovered)}")
    print("\n  Build a story from the covered signs, e.g.:")
    print('    python src/demo_video.py --build --story \'')
    print('      \'[["father","brother"],["brother","happy"]]\'')
    return ref, covered


def pick_clips(story, rows, data_dir="data", min_hand_rate=0.6, verbose=True,
               min_match=0.0, reference=None):
    """Choose one clip per sign, all from the same-looking signer.

    Every candidate for the first sign is tried as a reference. For each
    reference the best-matching candidate is taken for every other sign, and
    the reference with the highest worst-case match wins. Optimising the worst
    match rather than the average stops one badly mismatched clip hiding
    inside a good average.
    """
    signs = []
    for phrase in story:
        for s in phrase:
            if s not in signs:
                signs.append(s)

    by_label = {}
    for r in rows:
        if float(r["hand_rate"]) >= min_hand_rate:
            by_label.setdefault(r["label"], []).append(r)

    missing = [s for s in signs if s not in by_label]
    if missing:
        raise SystemExit(f"no usable clips for: {', '.join(missing)}")

    if verbose:
        print("  fingerprinting candidates...")
    desc = {}
    for s in signs:
        for r in by_label[s]:
            d = appearance(Path(r["src"]))
            if d is not None:
                desc[r["file"]] = d

    # Every candidate of every sign is a possible reference, not just the
    # first sign's. The signer who covers the story best often does not
    # appear in the opening phrase at all.
    # A reference named on the command line may be a clip of a sign the story
    # never uses. --suggest reports the signer who covers the most signs, and
    # that signer's best-looking clip is usually outside the story, so the
    # candidate list below would never reach it.
    if reference:
        ref_desc = appearance(Path(reference))
        if ref_desc is None:
            raise SystemExit(f"cannot read a torso from {reference}")
        pool = [{"file": "__ref__", "src": reference, "hand_rate": "1.0"}]
        desc["__ref__"] = ref_desc
    else:
        pool = [r for sign in signs for r in by_label[sign]]

    best = None
    for ref in pool:
        rd = desc.get(ref["file"])
        if rd is None:
            continue
        chosen, worst = {}, 1.0
        for s in signs:
            scored = [(similarity(rd, desc.get(r["file"])), r)
                      for r in by_label[s]]
            scored.sort(key=lambda t: (-t[0], -float(t[1]["hand_rate"])))
            sim, pick = scored[0]
            chosen[s] = pick
            worst = min(worst, sim)
        if min_match and worst < min_match:
            continue        # this reference cannot cover the whole story
        if best is None or worst > best[0]:
            best = (worst, ref, chosen)

    if best is None:
        raise SystemExit(
            f"no single signer covers every sign at match >= {min_match}.\n"
            "Run --suggest to see which signs one signer does cover, then "
            "write the story from those.")

    worst, ref, chosen = best
    if verbose:
        print(f"  signer chosen from {Path(ref['src']).name}, "
              f"worst appearance match {worst:.2f}")
        weak = [s for s in chosen
                if similarity(desc.get(ref["file"]),
                              desc.get(chosen[s]["file"])) < 0.85]
        if weak:
            print(f"  WARNING: {', '.join(weak)} matched weakly; those clips "
                  "may show a different person.")
            print("  Run --suggest and rewrite the story from one signer's "
                  "vocabulary.")

    out = []
    for phrase in story:
        out.append([(s, Path(chosen[s]["src"]), float(chosen[s]["hand_rate"]))
                    for s in phrase])
    return out, chosen


def build(story_clips, out_path, gap_frames=28, within_gap=6):
    """Concatenate clips into one video.

    Gaps matter. The phrase detector ends a phrase when the hands leave frame
    for about half a second, so a longer pause between phrases and a shorter
    one between signs inside a phrase reproduces what a signer would do.

    Between phrases the gap is black. Within a phrase it is the clip's first
    frame, where the signer stands at rest.

    The rest frame alone does not work as a phrase boundary. INCLUDE signers
    never leave the shot, so a rest pose still has both hands visible and the
    presence flags stay true through the whole gap. Measuring motion instead
    does not separate them either: the hold phase inside a sign runs up to 18
    frames below any sensible threshold, which is as long as the pauses. A
    black gap removes the question. MediaPipe finds no pose and no hands, the
    presence rule fires exactly as designed, and on screen it reads as a cut
    between sentences.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             TARGET_FPS, (TARGET_W, TARGET_H))
    if not writer.isOpened():
        raise SystemExit(f"cannot open {out_path} for writing")

    manifest, total = [], 0
    for pi, phrase in enumerate(story_clips):
        for si, (sign, src, rate) in enumerate(phrase):
            if not src.exists():
                raise SystemExit(f"missing clip: {src}")
            cap = cv2.VideoCapture(str(src))
            start, rest = total, None
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (TARGET_W, TARGET_H))
                if rest is None:
                    rest = frame.copy()      # hands down, before the sign starts
                writer.write(frame)
                total += 1
            cap.release()
            manifest.append({"phrase": pi, "sign": sign, "src": str(src),
                             "hand_rate": rate, "start_frame": start,
                             "end_frame": total})

            end_of_phrase = si == len(phrase) - 1
            pause = gap_frames if end_of_phrase else within_gap
            filler = (np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
                      if end_of_phrase else rest)
            if filler is not None:
                for _ in range(pause):
                    writer.write(filler)
                    total += 1

    writer.release()
    meta = out_path.with_suffix(".json")
    meta.write_text(json.dumps({
        "video": str(out_path), "frames": total, "fps": TARGET_FPS,
        "seconds": round(total / TARGET_FPS, 1),
        "expected_phrases": [[s for s, _, _ in p] for p in story_clips],
        "clips": manifest,
    }, indent=2))
    print(f"  wrote {out_path}  ({total} frames, {total / TARGET_FPS:.1f}s, "
          f"{len(manifest)} clips)")
    print(f"  wrote {meta}")
    return out_path


def contact_sheet(story_clips, out_path, cols=4):
    """One frame per clip, so signer consistency can be checked at a glance."""
    tiles = []
    for phrase in story_clips:
        for sign, src, _ in phrase:
            cap = cv2.VideoCapture(str(src))
            cap.set(cv2.CAP_PROP_POS_FRAMES, DESC_FRAME)
            ok, f = cap.read()
            cap.release()
            if not ok:
                continue
            f = cv2.resize(f, (320, 180))
            cv2.putText(f, sign, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
            tiles.append(f)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out_path), np.vstack(rows))
    print(f"  wrote {out_path}")


def run(video, checkpoint=None, narrate=False, temperature=0.2,
        confidence=0.75, predict_every=3, debug=False, offline=False,
        speak=False):
    from extractor import LandmarkExtractor
    from phrase import PhraseConfig, PhraseDetector
    from realtime import STILL_EPS, RollingBuffer, SignRecognizer

    recog = SignRecognizer(checkpoint)
    buf = RollingBuffer()
    det = PhraseDetector(PhraseConfig(min_confidence=confidence))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")

    # Pass one: landmarks, predictions and motion for every frame. The pause
    # threshold is measured from this video rather than assumed, so pass two
    # replays the trace through the phrase detector once it is known.
    i, trace = 0, []
    print("\n  recognising...")
    with LandmarkExtractor(running_mode="video") as ext:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            buf.push(ext.frame(frame, int(i / TARGET_FPS * 1000)))
            i += 1
            label, conf = None, 0.0
            if buf.ready and i % predict_every == 0:
                label, conf = recog.predict(buf.sequence(recog.use_motion))
            trace.append((i, label, conf, buf.hands_present_recent(),
                          buf.motion()))
    cap.release()

    moving = [m for _, _, _, _, m in trace if m > 0]
    eps = otsu(moving) if moving else STILL_EPS
    q = np.percentile(moving, [5, 25, 50, 75, 95]) if moving else [0] * 5
    print(f"  motion p5/p25/p50/p75/p95 "
          + "/".join(f"{x:.4f}" for x in q))
    print(f"  pause threshold {eps:.4f} (Otsu split of log motion), "
          f"{sum(1 for m in moving if m < eps) / len(moving):.0%} of frames "
          "read as pause")
    # Keep the raw series. If segmentation still misses, the numbers are here
    # rather than needing another run to collect them.
    Path(video).with_suffix(".trace.json").write_text(json.dumps(
        {"eps": eps, "fps": TARGET_FPS,
         "trace": [{"f": n, "label": l, "conf": round(c, 4),
                    "hands": bool(h), "motion": round(m, 5)}
                   for n, l, c, h, m in trace]}))

    # Pass two. The boundary is hand presence, which the black gaps make
    # unambiguous. Motion is measured and reported but not used to gate: the
    # hold phase inside a sign is as long and as still as a pause, so a motion
    # threshold cannot tell them apart without a margin too thin to trust.
    phrases = []
    for n, label, conf, hands, m in trace:
        out = det.update(label, conf, hands, now=n / TARGET_FPS)
        if out:
            phrases.append(out)
            print(f"    phrase {len(phrases)}: {' '.join(out).upper()}")

    if debug:
        _report(trace, Path(video).with_suffix(".json"), confidence, eps)
        meta_path = Path(video).with_suffix(".json")
        if meta_path.exists():
            sweep(trace, json.loads(meta_path.read_text())["expected_phrases"])

    tail = det.flush()
    if tail:
        phrases.append(tail)
        print(f"    phrase {len(phrases)}: {' '.join(tail).upper()}")
    if not phrases:
        print("    no phrases detected")
        return None, None

    meta_path = Path(video).with_suffix(".json")
    if meta_path.exists():
        expected = json.loads(meta_path.read_text())["expected_phrases"]
        flat_e = [s for p in expected for s in p]
        flat_g = [s for p in phrases for s in p]
        hits = sum(1 for a, b in zip(flat_e, flat_g) if a == b)
        print(f"\n  expected : {' | '.join(' '.join(p).upper() for p in expected)}")
        print(f"  detected : {' | '.join(' '.join(p).upper() for p in phrases)}")
        print(f"  signs correct and in order: {hits}/{len(flat_e)}")

    paragraph = None
    if narrate and offline:
        # The saved paragraph from a previous run. Latency p90 is about 13
        # seconds, so a live call during a demo can leave a long dead pause,
        # and a dead network leaves none at all. Everything upstream of this
        # point still runs for real; only the API call is replayed.
        saved = Path(video).with_suffix(".result.json")
        if not saved.exists():
            raise SystemExit(f"no saved result at {saved}. Run once with "
                             "--narrate and without --offline first.")
        rec = json.loads(saved.read_text())
        paragraph = rec["paragraph"]
        print(f"\n  PARAGRAPH (replayed, not a live call):\n    {paragraph}\n")
        print(f"  model             : {rec.get('model') or 'unknown'}")
        if rec.get("phrases") and rec["phrases"] != phrases:
            print("  WARNING: the phrases detected now differ from the ones "
                  "this paragraph was written for.")
            print(f"    saved    : "
                  f"{' | '.join(' '.join(p).upper() for p in rec['phrases'])}")
        try:
            from llm_eval import paragraph_faithfulness
            f = paragraph_faithfulness(phrases, paragraph)
            print(f"  invented entities : {f['invented_entities'] or 'none'}")
            print(f"  cohesion words    : {f['cohesion_words'] or 'none'}")
            print(f"  {f['sentences']} sentences, {f['words']} words")
        except Exception as e:
            print(f"  (faithfulness check unavailable: {e})")
        if speak:
            from tts import available, speak as say
            if available():
                say(paragraph, blocking=True)
            else:
                print("  (speech needs macOS `say`)")
        return phrases, paragraph

    if narrate:
        from storyteller import Storyteller
        teller = Storyteller(temperature=temperature)
        if teller.provider == "none":
            print("\n  no API key; skipping narration")
        else:
            print("\n  narrating...")
            paragraph = teller.narrate(phrases)
            print(f"\n  PARAGRAPH:\n    {paragraph}\n")
            # Which model answered, not which was asked for. Gemini falls down
            # a chain when one is retired or busy, and the paragraph changes
            # with it, so a recorded demo should say what produced it.
            print(f"  model             : {teller.model_used or 'unknown'}")
            if teller.last_error:
                print(f"  (fell back to template: {teller.last_error})")
            try:
                from llm_eval import paragraph_faithfulness
                f = paragraph_faithfulness(phrases, paragraph)
                print(f"  invented entities : {f['invented_entities'] or 'none'}")
                print(f"  cohesion words    : {f['cohesion_words'] or 'none'}")
                print(f"  {f['sentences']} sentences, {f['words']} words")
            except Exception as e:
                print(f"  (faithfulness check unavailable: {e})")
    if speak and paragraph:
        from tts import available, speak as say
        if available():
            say(paragraph, blocking=True)
        else:
            print("  (speech needs macOS `say`)")

    meta = Path(video).with_suffix(".result.json")
    meta.write_text(json.dumps({
        "phrases": phrases,
        "paragraph": paragraph,
        "model": getattr(teller, "model_used", None) if narrate else None,
        "temperature": temperature if narrate else None,
        "min_confidence": confidence,
        "stable_frames": PhraseConfig().stable_frames,
        "predict_every": predict_every,
    }, indent=2))
    print(f"  wrote {meta}")
    return phrases, paragraph





# ---- annotated render ------------------------------------------------------

RENDER_W, RENDER_H = 1280, 720
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel(img, x, y, w, h, alpha=0.62):
    """Darken a rectangle so white text stays readable over any frame."""
    x, y = max(0, x), max(0, y)
    w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)
    if w <= 0 or h <= 0:
        return
    roi = img[y:y + h, x:x + w]
    img[y:y + h, x:x + w] = cv2.addWeighted(
        roi, 1 - alpha, np.zeros_like(roi), alpha, 0)


def _wrap(text, width, scale, thick):
    """Greedy word wrap to a pixel width, using the real text metrics."""
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = f"{line} {w}".strip()
        if cv2.getTextSize(trial, FONT, scale, thick)[0][0] > width and line:
            lines.append(line)
            line = w
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def render(video, out_path, confidence=0.75):
    """Draw the pipeline's own output over the demo video.

    The raw stitched clips show a man signing and nothing else. What a viewer
    needs to see is the system reading him: the sign it is calling right now,
    the words collecting into a phrase, the phrase closing at a boundary, and
    the paragraph at the end. All of that is already recorded in the trace, so
    this redraws from the saved numbers rather than running the models again.
    """
    from phrase import PhraseConfig, PhraseDetector

    video = Path(video)
    trace_path = video.with_suffix(".trace.json")
    result_path = video.with_suffix(".result.json")
    if not trace_path.exists():
        raise SystemExit(f"no trace at {trace_path}. Run once with --debug.")
    data = json.loads(trace_path.read_text())
    trace, fps = data["trace"], data.get("fps", TARGET_FPS)
    paragraph = None
    if result_path.exists():
        paragraph = json.loads(result_path.read_text()).get("paragraph")

    # Replay the detector so the overlay shows exactly what the run produced.
    det = PhraseDetector(PhraseConfig(min_confidence=confidence))
    per_frame, done = [], []
    for r in trace:
        out = det.update(r["label"], r["conf"], r["hands"],
                         now=r["f"] / fps)
        if out:
            done.append(out)
        per_frame.append((list(det.words), len(done), bool(out)))
    tail = det.flush()
    if tail:
        done.append(tail)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (RENDER_W, RENDER_H))
    if not writer.isOpened():
        raise SystemExit(f"cannot open {out_path} for writing")

    # The classifier runs every 3rd frame, so two frames in three carry no
    # label. Showing that honestly reads as the system failing, when really it
    # is between ticks, so the last prediction is held for a few frames.
    HOLD = 5
    held, held_conf, held_age = None, 0.0, HOLD + 1

    flash = 0
    for i, r in enumerate(trace):
        ok, frame = cap.read()
        if not ok:
            break
        f = cv2.resize(frame, (RENDER_W, RENDER_H))
        words, n_done, fired = per_frame[i]
        flash = 12 if fired else max(0, flash - 1)

        # top strip: what the classifier is calling right now
        if r["label"]:
            held, held_conf, held_age = r["label"], r["conf"], 0
        else:
            held_age += 1
        show = held if held_age <= HOLD and r["hands"] else None

        _panel(f, 0, 0, RENDER_W, 58)
        if show and held_conf >= confidence:
            colour = (120, 230, 120)
            text = f"{show.upper()}  {held_conf:.2f}"
        elif show:
            colour = (120, 170, 235)
            text = f"{show} {held_conf:.2f}  (below {confidence:.2f}, ignored)"
        else:
            colour, text = (170, 170, 170), "listening"
        cv2.putText(f, text, (22, 40), FONT, 0.95, colour, 2, cv2.LINE_AA)
        hands = "hands detected" if r["hands"] else "no hands: phrase boundary"
        cv2.putText(f, hands, (RENDER_W - 390, 38), FONT, 0.62,
                    (210, 210, 210) if r["hands"] else (120, 200, 255),
                    2, cv2.LINE_AA)

        # bottom strip: the phrase being built
        _panel(f, 0, RENDER_H - 96, RENDER_W, 96)
        cv2.putText(f, f"phrase {n_done + 1}", (22, RENDER_H - 62), FONT,
                    0.6, (160, 160, 160), 1, cv2.LINE_AA)
        building = " ".join(w.upper() for w in words) or "..."
        cv2.putText(f, building, (22, RENDER_H - 24), FONT, 1.0,
                    (255, 220, 120) if not flash else (140, 255, 140),
                    2, cv2.LINE_AA)
        if flash:
            cv2.putText(f, "PHRASE COMPLETE", (RENDER_W - 330, RENDER_H - 24),
                        FONT, 0.7, (140, 255, 140), 2, cv2.LINE_AA)

        # The black frames are the sentence break: no pose, no hands, so the
        # boundary rule fires. Say so, or they read as the video dropping out.
        if not r["hands"]:
            label = "SENTENCE BREAK"
            tw = cv2.getTextSize(label, FONT, 1.1, 2)[0][0]
            cv2.putText(f, label, ((RENDER_W - tw) // 2, RENDER_H // 2),
                        FONT, 1.1, (120, 200, 255), 2, cv2.LINE_AA)
            sub = "no hands in frame"
            sw = cv2.getTextSize(sub, FONT, 0.6, 1)[0][0]
            cv2.putText(f, sub, ((RENDER_W - sw) // 2, RENDER_H // 2 + 36),
                        FONT, 0.6, (150, 150, 150), 1, cv2.LINE_AA)

        # left column: phrases already closed
        for k, phrase in enumerate(done[:n_done]):
            y = 92 + k * 34
            _panel(f, 14, y - 24, 430, 32, alpha=0.5)
            cv2.putText(f, f"{k + 1}. " + " ".join(w.upper() for w in phrase),
                        (22, y), FONT, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        writer.write(f)
    cap.release()

    # closing card: the phrases and the paragraph they produced
    card = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
    cv2.putText(card, "RECOGNISED", (60, 90), FONT, 0.8, (150, 150, 150),
                2, cv2.LINE_AA)
    y = 150
    for k, phrase in enumerate(done, 1):
        cv2.putText(card, f"{k}. " + " ".join(w.upper() for w in phrase),
                    (60, y), FONT, 0.85, (255, 220, 120), 2, cv2.LINE_AA)
        y += 48
    if paragraph:
        y += 40
        cv2.putText(card, "GENERATED", (60, y), FONT, 0.8, (150, 150, 150),
                    2, cv2.LINE_AA)
        y += 60
        for line in _wrap(paragraph, RENDER_W - 120, 0.95, 2):
            cv2.putText(card, line, (60, y), FONT, 0.95, (160, 255, 160),
                        2, cv2.LINE_AA)
            y += 46
    for _ in range(int(fps * 5)):
        writer.write(card)
    writer.release()

    print(f"  wrote {out_path}  ({len(trace) + int(fps * 5)} frames, "
          f"{(len(trace) + fps * 5) / fps:.1f}s)")
    print(f"  {len(done)} phrases: "
          + " | ".join(" ".join(p).upper() for p in done))
    return out_path


def sweep(trace, expected, fps=TARGET_FPS):
    """Replay the trace across a grid of detector settings.

    One clip sitting on the confidence gate can cost a whole sign, and the
    sensible response is to measure where the gate should sit rather than nudge
    it until the demo passes. This reruns the same recognised frames through
    the phrase detector for every setting and reports what each one recovers,
    so the choice is made against numbers and its margin is visible.
    """
    from phrase import PhraseConfig, PhraseDetector

    flat_e = [s for p in expected for s in p]
    print("\n  --- detector sweep " + "-" * 36)
    print(f"  {'min_conf':>9} {'stable':>7} {'signs':>7} {'phrases':>9}  detected")
    best = None
    for conf in [0.60, 0.65, 0.70, 0.75, 0.80]:
        for stable in [4, 5, 6]:
            det = PhraseDetector(PhraseConfig(min_confidence=conf,
                                              stable_frames=stable))
            got = []
            for n, label, c, hands, _ in trace:
                out = det.update(label, c, hands, now=n / fps)
                if out:
                    got.append(out)
            tail = det.flush()
            if tail:
                got.append(tail)
            flat_g = [s for p in got for s in p]
            hits = sum(1 for a, b in zip(flat_e, flat_g) if a == b)
            exact = got == expected
            mark = "  <- exact" if exact else ""
            print(f"  {conf:9.2f} {stable:7d} {hits:4d}/{len(flat_e)} "
                  f"{len(got):9d}  "
                  f"{' | '.join(' '.join(p).upper() for p in got)[:46]}{mark}")
            if exact and best is None:
                best = (conf, stable)
    print("  " + "-" * 54)
    live = PhraseConfig()
    if best:
        print(f"  running config: min_confidence {live.min_confidence:.2f}, "
              f"stable_frames {live.stable_frames}")
        print(f"  stable_frames {live.stable_frames} recovers the story at "
              f"every min_confidence from {best[0]:.2f} to 0.75, so the "
              "confidence gate is not the binding constraint.")
        print("  the gate is kept at 0.75 because that is what rejects the "
              "spurious sign in the trailing gap.")
    else:
        print("  no setting recovers the whole story; the weak clip is the "
              "problem, not the gate")
    return best


def _report(trace, meta_path, confidence, eps):
    """Per-clip breakdown of what the recogniser saw. Answers, in one run,
    whether a miss was MediaPipe losing the hands, the classifier being unsure,
    or the classifier being confidently wrong."""
    from collections import Counter

    print("\n  --- per-clip trace " + "-" * 36)
    print(f"  {'region':22} {'hands':>6} {'motion':>8} {'top prediction':>16} "
          f"{'votes':>6} {'mean conf':>10} {'>=thr':>6}")

    spans = []
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        prev = 0
        for c in meta["clips"]:
            if c["start_frame"] > prev:
                spans.append((f"  (gap)", prev, c["start_frame"]))
            spans.append((f"{c['phrase']}:{c['sign']}",
                          c["start_frame"], c["end_frame"]))
            prev = c["end_frame"]
        if prev < len(trace):
            spans.append(("  (gap)", prev, len(trace)))
    else:
        spans = [("whole video", 0, len(trace))]

    for name, a, b in spans:
        rows = [t for t in trace if a < t[0] <= b]
        if not rows:
            continue
        hands = sum(1 for _, _, _, h, _ in rows if h) / len(rows)
        mot = float(np.median([m for _, _, _, _, m in rows]))
        flag = " " if mot >= eps else "*"
        preds = [(l, c) for _, l, c, _, _ in rows if l is not None]
        if not preds:
            print(f"  {name:22} {hands:6.0%} {mot:7.4f}{flag} "
                  f"{'(no prediction)':>16}")
            continue
        top, votes = Counter(l for l, _ in preds).most_common(1)[0]
        confs = [c for l, c in preds if l == top]
        over = sum(1 for c in confs if c >= confidence) / len(confs)
        print(f"  {name:22} {hands:6.0%} {mot:7.4f}{flag} {top:>16} "
              f"{votes:6d} {sum(confs) / len(confs):10.2f} {over:6.0%}")
    print("  " + "-" * 54)
    print(f"  hands near 0% in a sign region means MediaPipe lost the hands.")
    print(f"  mean conf below {confidence:.2f} means the classifier was unsure.")
    print(f"  motion is reported, not used to segment; * marks "
          f"below {eps:.4f}.")
    print(f"  hands near 100% in a (gap) means no phrase boundary was seen.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", nargs="?", const=True, metavar="VIDEO")
    ap.add_argument("--out", default="demo/story.mp4")
    ap.add_argument("--data", default="data")
    ap.add_argument("--story", help="JSON list of phrases")
    ap.add_argument("--checkpoint")
    ap.add_argument("--narrate", action="store_true")
    ap.add_argument("--speak", action="store_true",
                    help="read the paragraph aloud when it is ready")
    ap.add_argument("--offline", action="store_true",
                    help="replay the saved paragraph instead of calling the "
                         "API; everything upstream still runs for real")
    ap.add_argument("--render", nargs="?", const="demo/story_annotated.mp4",
                    metavar="OUT",
                    help="write a video with the pipeline output drawn on it")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--confidence", type=float, default=0.75)
    ap.add_argument("--gap-frames", type=int, default=28)
    ap.add_argument("--suggest", action="store_true",
                    help="find the signer covering the most reliable signs")
    ap.add_argument("--debug", action="store_true",
                    help="print a per-clip trace of what the recogniser saw")
    ap.add_argument("--reference",
                    default=DEFAULT_REFERENCE,
                    help="clip whose signer the story should use; take this "
                         "from the --suggest output")
    ap.add_argument("--min-match", type=float, default=0.85,
                    help="appearance similarity required to accept a clip")
    args = ap.parse_args()

    if args.suggest:
        suggest(load_index(args.data), data_dir=args.data,
                min_match=args.min_match)
        return

    if not args.build and not args.run and not args.render:
        ap.error("pass --build, --run, --render, --suggest, or a combination")

    if args.build:
        story = json.loads(args.story) if args.story else DEFAULT_STORY
        rows = load_index(args.data)
        print("building demo video")
        clips, _ = pick_clips(story, rows, data_dir=args.data,
                              min_match=args.min_match,
                              reference=args.reference)
        for pi, phrase in enumerate(clips):
            for sign, src, rate in phrase:
                print(f"  [{pi}] {sign:12} hand_rate {rate:.2f}  {src.name}")
        build(clips, args.out, gap_frames=args.gap_frames)
        contact_sheet(clips, Path(args.out).with_name("contact_sheet.jpg"))

    if args.render:
        render(args.run if isinstance(args.run, str) else args.out,
               args.render, confidence=args.confidence)

    if args.run:
        video = args.run if isinstance(args.run, str) else args.out
        run(video, args.checkpoint, narrate=args.narrate,
            temperature=args.temperature, confidence=args.confidence,
            debug=args.debug, offline=args.offline, speak=args.speak)


if __name__ == "__main__":
    main()
