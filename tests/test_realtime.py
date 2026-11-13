"""Tests for the live-recognition buffer. No camera, no torch, no mediapipe.

Run with: python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import BUFFER_LEN, FEATURE_DIM, SEQ_LEN  # noqa: E402
from realtime import RollingBuffer  # noqa: E402


def frame(left=1.0, right=1.0, pose=1.0, seed=None):
    rng = np.random.default_rng(seed)
    v = rng.random(FEATURE_DIM).astype(np.float32)
    v[-3], v[-2], v[-1] = pose, left, right
    return v


def test_not_ready_until_enough_frames():
    b = RollingBuffer()
    for i in range(SEQ_LEN - 1):
        b.push(frame(seed=i))
        assert not b.ready
    b.push(frame(seed=99))
    assert b.ready


def test_window_is_capped():
    b = RollingBuffer()
    for i in range(BUFFER_LEN + 40):
        b.push(frame(seed=i))
    assert len(b.frames) == BUFFER_LEN


def test_sequence_shape_matches_training():
    b = RollingBuffer()
    for i in range(BUFFER_LEN):
        b.push(frame(seed=i))
    assert b.sequence(with_motion=False).shape == (SEQ_LEN, FEATURE_DIM)
    assert b.sequence(with_motion=True).shape == (SEQ_LEN, FEATURE_DIM * 2)


def test_sequence_resamples_a_partial_window():
    b = RollingBuffer(min_fill=4)
    for i in range(6):
        b.push(frame(seed=i))
    assert b.ready
    assert b.sequence().shape == (SEQ_LEN, FEATURE_DIM * 2)


def test_wrong_feature_width_is_rejected():
    b = RollingBuffer()
    with pytest.raises(ValueError):
        b.push(np.zeros(FEATURE_DIM - 1, dtype=np.float32))


def test_empty_buffer_has_no_hands():
    b = RollingBuffer()
    assert b.hands_present is False
    assert b.hands_present_recent() is False


def test_hands_present_reads_the_last_frame():
    b = RollingBuffer()
    b.push(frame(left=1.0, right=1.0))
    assert b.hands_present
    b.push(frame(left=0.0, right=0.0))
    assert not b.hands_present


def test_recent_tolerates_a_single_dropout():
    """One missed detection mid-sign must not look like hands down."""
    b = RollingBuffer()
    b.push(frame(left=1.0, right=1.0))
    b.push(frame(left=0.0, right=0.0))
    assert b.hands_present_recent(n=3)
    b.push(frame(left=0.0, right=0.0))
    b.push(frame(left=0.0, right=0.0))
    assert not b.hands_present_recent(n=3)


def test_one_hand_counts_as_present():
    b = RollingBuffer()
    b.push(frame(left=1.0, right=0.0))
    assert b.hands_present


def test_clear_resets():
    b = RollingBuffer()
    for i in range(SEQ_LEN):
        b.push(frame(seed=i))
    b.clear()
    assert not b.ready and not b.hands_present


def test_empty_sequence_raises():
    with pytest.raises(ValueError):
        RollingBuffer().sequence()


def test_first_motion_delta_is_zero():
    """Same invariant the training features have; the paths must agree."""
    b = RollingBuffer()
    for i in range(BUFFER_LEN):
        b.push(frame(seed=i))
    seq = b.sequence(with_motion=True)
    assert np.abs(seq[0, FEATURE_DIM:]).max() == 0.0


# ---- phrase detector: skipped frames vs rejected predictions ---------------
# The callers run the classifier every Nth frame and pass label=None on the
# rest. Treating that None as evidence against the candidate reset the counter
# twice for every time it advanced, so stable_frames was unreachable and a
# whole video produced no phrases at all.

from phrase import PhraseConfig, PhraseDetector  # noqa: E402


def _det(**kw):
    return PhraseDetector(PhraseConfig(**kw))


def test_skipped_frames_do_not_reset_the_candidate():
    d = _det(stable_frames=5)
    t = 0.0
    for _ in range(5):                 # one prediction, two skips, repeated
        d.update("cow", 0.9, True, now=t); t += 0.04
        d.update(None, 0.0, True, now=t); t += 0.04
        d.update(None, 0.0, True, now=t); t += 0.04
    assert d.words == ["cow"]


def test_low_confidence_still_resets_the_candidate():
    d = _det(stable_frames=3, min_confidence=0.75)
    t = 0.0
    d.update("cow", 0.9, True, now=t); t += 0.04
    d.update("cow", 0.9, True, now=t); t += 0.04
    d.update("cow", 0.10, True, now=t); t += 0.04   # a real, rejected guess
    d.update("cow", 0.9, True, now=t); t += 0.04
    d.update("cow", 0.9, True, now=t); t += 0.04
    assert d.words == []               # the counter restarted, so no word yet
    d.update("cow", 0.9, True, now=t)
    assert d.words == ["cow"]


def test_hands_down_still_clears_a_half_formed_candidate():
    d = _det(stable_frames=5)
    t = 0.0
    for _ in range(4):
        d.update("cow", 0.9, True, now=t); t += 0.04
    d.update(None, 0.0, False, now=t); t += 0.04     # hands drop mid-sign
    for _ in range(4):
        d.update("cow", 0.9, True, now=t); t += 0.04
    assert d.words == []


def test_a_realistic_stream_emits_one_phrase_per_rest_gap():
    """Three signs, a short gap between each, a long gap at the end."""
    d = _det(stable_frames=5, hands_down_frames=12)
    t, out = 0.0, []
    for sign in ["father", "cow", "window"]:
        for j in range(30):                     # predict every 3rd frame
            lab = sign if j % 3 == 0 else None
            r = d.update(lab, 0.9, True, now=t); t += 0.04
            if r:
                out.append(r)
        for _ in range(6):                      # short rest, within a phrase
            r = d.update(None, 0.0, False, now=t); t += 0.04
            if r:
                out.append(r)
    for _ in range(20):                         # long rest, ends the phrase
        r = d.update(None, 0.0, False, now=t); t += 0.04
        if r:
            out.append(r)
    assert out == [["father", "cow", "window"]]


# ---- pause detection on stitched video ------------------------------------
# A rest pose still has both hands in shot, so presence cannot mark a phrase
# boundary on dataset clips. A repeated frame has no motion, and that can.

def test_static_needs_a_full_window_of_identical_frames():
    b = RollingBuffer()
    for i in range(SEQ_LEN):
        b.push(frame(seed=i))
    assert not b.is_static(n=6)
    still = frame(seed=1234)
    for _ in range(6):                 # six repeats: the window still spans a change
        b.push(still.copy())
    assert not b.is_static(n=6)
    b.push(still.copy())               # seventh makes the whole window identical
    assert b.is_static(n=6)


def test_motion_clears_static_again():
    b = RollingBuffer()
    still = frame(seed=7)
    for _ in range(10):
        b.push(still.copy())
    assert b.is_static(n=6)
    b.push(frame(seed=8))
    assert not b.is_static(n=6)


def test_within_phrase_gap_does_not_end_a_phrase():
    """6 repeated frames separate signs; that must not close the phrase."""
    b = RollingBuffer()
    d = _det(stable_frames=5, hands_down_frames=12)
    t, out = 0.0, []
    for sign in ["father", "cow"]:
        live = [frame(seed=hash(sign) % 1000 + j) for j in range(30)]
        for j, f in enumerate(live):
            b.push(f)
            lab = sign if j % 3 == 0 else None
            r = d.update(lab, 0.9,
                         b.hands_present_recent() and not b.is_static(n=6),
                         now=t)
            t += 0.04
            if r:
                out.append(r)
        rest = live[0]
        for _ in range(6):
            b.push(rest.copy())
            r = d.update(None, 0.0,
                         b.hands_present_recent() and not b.is_static(n=6),
                         now=t)
            t += 0.04
            if r:
                out.append(r)
    assert out == []
    assert d.words == ["father", "cow"]


def test_between_phrase_gap_ends_the_phrase():
    """28 repeated frames must close it, with margin over hands_down_frames."""
    b = RollingBuffer()
    d = _det(stable_frames=5, hands_down_frames=12)
    t, out = 0.0, []
    live = [frame(seed=j) for j in range(30)]
    for j, f in enumerate(live):
        b.push(f)
        r = d.update("father" if j % 3 == 0 else None, 0.9,
                     b.hands_present_recent() and not b.is_static(n=6), now=t)
        t += 0.04
        if r:
            out.append(r)
    rest = live[0]
    inactive = 0
    for _ in range(28):
        b.push(rest.copy())
        active = b.hands_present_recent() and not b.is_static(n=6)
        inactive += not active
        r = d.update(None, 0.0, active, now=t)
        t += 0.04
        if r:
            out.append(r)
    assert inactive >= 12 + 6, f"only {inactive} inactive frames, too tight"
    assert out == [["father"]]


# ---- Otsu pause threshold -------------------------------------------------

def test_otsu_survives_a_right_skewed_signing_cluster():
    """The failure that was actually seen: on a linear axis the long
    high-motion tail drags the cut up into the signing cluster."""
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
    from demo_video import otsu

    rng = np.random.default_rng(0)
    pause = np.exp(rng.normal(np.log(0.012), 0.45, 200))
    signing = np.exp(rng.normal(np.log(0.090), 0.55, 600))   # long right tail
    vals = np.concatenate([pause, signing])

    t = otsu(vals, log=True)
    assert (pause < t).mean() > 0.9, f"threshold {t:.4f} misses pauses"
    assert (signing >= t).mean() > 0.9, f"threshold {t:.4f} eats signing"


def test_motion_is_zero_until_the_window_fills():
    b = RollingBuffer()
    assert b.motion(n=6) == 0.0
    for i in range(4):
        b.push(frame(seed=i))
    assert b.motion(n=6) == 0.0


def test_motion_separates_a_frozen_frame_from_real_movement():
    b = RollingBuffer()
    still = frame(seed=3)
    for _ in range(10):
        b.push(still.copy())
    frozen = b.motion(n=6)
    for i in range(10):
        b.push(frame(seed=100 + i))
    assert frozen < b.motion(n=6) / 10


def test_default_stable_frames_tolerates_one_dropped_detection():
    """A single lost hand-detection frame mid-sign must not cost the word.

    At stable_frames=5 with predict_every=3 it did, which is how t-shirt
    vanished from the demo video.
    """
    d = PhraseDetector(PhraseConfig())
    t = 0.0
    for j in range(24):
        hands = j != 7                      # one dropped frame, mid-sign
        d.update("t-shirt" if j % 3 == 0 else None, 0.78, hands, now=t)
        t += 0.04
    assert d.words == ["t-shirt"]
