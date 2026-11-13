"""Run with: python -m pytest tests -q"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phrase import PhraseConfig, PhraseDetector  # noqa: E402
from storyteller import (  # noqa: E402
    Storyteller, content_words, fallback_sentence, gloss_tokens, invented_words,
)

CFG = PhraseConfig(min_confidence=0.75, stable_frames=3, hands_down_frames=4,
                   idle_seconds=4.0, max_words=8, repeat_block_seconds=1.5)


def feed(det, label, conf, hands, n=1, t=0.0):
    """Push n identical frames, return the last non-None result."""
    out = None
    for i in range(n):
        r = det.update(label, conf, hands, now=t + i * 0.04)
        if r is not None:
            out = r
    return out


# ---- phrase detection -------------------------------------------------

def test_stable_run_becomes_one_word():
    d = PhraseDetector(CFG)
    feed(d, "hello", 0.9, True, n=30, t=0.0)
    assert d.words == ["hello"]      # 30 frames of one gesture, one word


def test_low_confidence_ignored():
    d = PhraseDetector(CFG)
    feed(d, "hello", 0.5, True, n=30, t=0.0)
    assert d.words == []


def test_hands_down_emits_phrase():
    d = PhraseDetector(CFG)
    feed(d, "i", 0.9, True, n=5, t=0.0)
    feed(d, "happy", 0.9, True, n=5, t=2.0)
    assert d.words == ["i", "happy"]
    phrase = feed(d, None, 0.0, False, n=4, t=3.0)
    assert phrase == ["i", "happy"]
    assert d.words == []             # detector reset after emitting


def test_hands_down_with_nothing_buffered_emits_nothing():
    d = PhraseDetector(CFG)
    assert feed(d, None, 0.0, False, n=20, t=0.0) is None


def test_distinct_signs_both_captured():
    d = PhraseDetector(CFG)
    feed(d, "father", 0.9, True, n=4, t=0.0)
    feed(d, "happy", 0.9, True, n=4, t=2.0)
    assert d.words == ["father", "happy"]


def test_repeat_blocked_then_allowed():
    d = PhraseDetector(CFG)
    feed(d, "dog", 0.9, True, n=4, t=0.0)
    feed(d, "dog", 0.9, True, n=4, t=0.5)     # inside the block window
    assert d.words == ["dog"]
    feed(d, "cow", 0.9, True, n=4, t=3.0)
    feed(d, "dog", 0.9, True, n=4, t=5.0)     # well past it
    assert d.words == ["dog", "cow", "dog"]


def test_max_words_forces_emit():
    d = PhraseDetector(PhraseConfig(stable_frames=2, max_words=3,
                                    repeat_block_seconds=0.0))
    feed(d, "i", 0.9, True, n=3, t=0.0)
    feed(d, "happy", 0.9, True, n=3, t=1.0)
    phrase = feed(d, "good", 0.9, True, n=3, t=2.0)
    assert phrase == ["i", "happy", "good"]


def test_idle_timeout_emits():
    d = PhraseDetector(CFG)
    feed(d, "hello", 0.9, True, n=4, t=0.0)
    phrase = d.update("hello", 0.9, True, now=10.0)
    assert phrase == ["hello"]


def test_flush_returns_partial():
    d = PhraseDetector(CFG)
    feed(d, "hello", 0.9, True, n=4, t=0.0)
    assert d.flush() == ["hello"]
    assert d.flush() is None


# ---- the constraint ---------------------------------------------------

def test_connectives_are_not_content():
    assert content_words("The dog is at the house.") == ["dog", "house"]


def test_pronouns_count_as_content():
    # "i", "it" and "you" are signs in the vocabulary, so they must be signed
    # rather than supplied by the model.
    assert content_words("I am happy.") == ["i", "happy"]
    assert "it" in invented_words(["dog", "black"], "It is black.")


def test_gloss_licenses_its_parts():
    tokens = gloss_tokens(["store_or_shop", "you_(plural)"])
    assert "store" in tokens and "shop" in tokens and "plural" in tokens


def test_clean_sentence_has_no_invented_words():
    assert invented_words(["i", "happy"], "I am happy.") == []
    assert invented_words(["brother", "dog", "black"],
                          "My brother's dog is black.") == []


def test_invented_entity_is_caught():
    bad = invented_words(["i", "happy"], "I am happy at the market.")
    assert "market" in bad


def test_plural_and_singular_both_allowed():
    assert invented_words(["dog"], "The dogs are big.") == ["big"]
    assert invented_words(["dog", "big_large"], "The dogs are big.") == []


# ---- fallback ---------------------------------------------------------

def test_fallback_uses_only_given_words():
    s = fallback_sentence(["good_morning", "teacher"])
    assert s == "Good morning, teacher."
    assert invented_words(["good_morning", "teacher"], s) == []


def test_fallback_on_missing_key():
    st = Storyteller(provider="none")
    sentence = st.add(["i", "happy"])
    assert sentence == "I, happy."          # template, not model output
    assert st.last_error is not None        # failure recorded, not hidden
    assert st.story == "I, happy."


def test_empty_phrase_adds_nothing():
    st = Storyteller(provider="none")
    assert st.add([]) is None
    assert st.sentences == []


# ---- reporting hygiene ----------------------------------------------------

def test_references_file_declares_its_provenance():
    """BLEU against references written alongside the system measures agreement
    with their author, not translation quality. The file has to say which it
    is, and the default must not be the reportable one."""
    import json as _json
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    data = _json.loads((root / "data" / "references.json").read_text())
    assert "provenance" in data, "references.json must declare provenance"
    assert data["provenance"] in ("draft", "human")


def test_load_refs_defaults_to_unreportable():
    import json as _json
    import tempfile
    from pathlib import Path as _P
    from eval_llm import load_refs
    with tempfile.TemporaryDirectory() as d:
        f = _P(d) / "r.json"
        f.write_text(_json.dumps({"single": [], "stories": []}))
        _, _, prov = load_refs(f)
    assert not str(prov).lower().startswith("human")


def test_storyteller_reports_the_model_that_answered():
    """Gemini walks a fallback chain, so the requested model and the answering
    model differ. Recording the requested one mislabels the results."""
    import storyteller as st_mod
    from storyteller import Storyteller

    teller = Storyteller(provider="gemini", model="asked-for")
    assert teller.model_used is None

    calls = {}

    def fake(provider, model, system, user, temperature):
        calls["model"] = model
        st_mod._set_served("actually-answered")
        return "Father is at the window."

    original = st_mod._call_api
    st_mod._call_api = fake
    try:
        teller.add(["father", "window"])
    finally:
        st_mod._call_api = original

    assert calls["model"] == "asked-for"
    assert teller.model_used == "actually-answered"
