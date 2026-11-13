"""Decide when a run of recognised signs is a finished phrase.

The classifier fires on every buffer, roughly 25 times a second. Most of those
predictions are the same sign repeated, or noise between signs. This module
turns that stream into discrete words, and decides when enough words have
accumulated to send to the language model.

Boundary rule: hands leaving the frame. Signers drop their hands between
utterances, and the presence flags in the feature vector already measure it, so
it costs no vocabulary. Two backstops catch the cases where hands never drop:
a gap with no new sign, and a hard cap on phrase length.
"""

import time
from dataclasses import dataclass, field


@dataclass
class PhraseConfig:
    min_confidence: float = 0.75      # below this, a prediction is discarded
    stable_frames: int = 4            # same sign this many times before accepting
    # 4, not 5, and measured rather than picked. The classifier runs every 3rd
    # frame, so 4 is about half a second of agreement. At 5 a clip only has to
    # drop one frame of hand detection mid-sign to restart the count, and the
    # word is lost: on the demo video that cost t-shirt entirely. A sweep over
    # min_confidence 0.60 to 0.80 recovered the whole story at 4 across the
    # whole range and at no value of min_confidence when set to 5, so this is
    # the binding constraint and the confidence gate is not.
    hands_down_frames: int = 12       # ~0.5s at 25fps, ends the phrase
    idle_seconds: float = 4.0         # no new sign for this long, ends the phrase
    max_words: int = 8                # hard cap, keeps API calls bounded
    repeat_block_seconds: float = 1.5  # ignore the same word repeated within this


@dataclass
class PhraseDetector:
    cfg: PhraseConfig = field(default_factory=PhraseConfig)

    words: list = field(default_factory=list)
    _candidate: str = None
    _candidate_count: int = 0
    _hands_down: int = 0
    _last_word_at: float = 0.0
    _last_word: str = None

    def reset(self):
        self.words = []
        self._candidate = None
        self._candidate_count = 0
        self._hands_down = 0
        self._last_word_at = 0.0
        self._last_word = None

    def update(self, label, confidence, hands_present, now=None):
        """Feed one classifier output. Returns a finished phrase or None.

        label          : predicted sign, or None if nothing was predicted
        confidence     : softmax probability for that label
        hands_present  : True if either hand was detected this frame
        now            : timestamp override, for tests
        """
        now = time.time() if now is None else now

        if hands_present:
            self._hands_down = 0
            self._accumulate(label, confidence, now)
        else:
            self._hands_down += 1
            # a dropped hand also invalidates any half-formed candidate
            self._candidate, self._candidate_count = None, 0
            if self._hands_down >= self.cfg.hands_down_frames and self.words:
                return self._emit()

        if self.words:
            if len(self.words) >= self.cfg.max_words:
                return self._emit()
            if now - self._last_word_at >= self.cfg.idle_seconds:
                return self._emit()
        return None

    def _accumulate(self, label, confidence, now):
        # None and a rejected prediction are not the same thing. The callers
        # run the classifier every Nth frame and pass None on the rest, so
        # treating None as evidence against the candidate reset the counter
        # twice for every time it advanced, and a run could never reach
        # stable_frames. None means "no prediction this frame", so it leaves
        # the candidate alone. A real prediction below threshold still clears
        # it, which is the rejection this guard was written for.
        if label is None:
            return
        if confidence < self.cfg.min_confidence:
            self._candidate, self._candidate_count = None, 0
            return

        if label == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = label, 1

        if self._candidate_count != self.cfg.stable_frames:
            return   # fires exactly once per stable run, not on every frame after

        # One continuous gesture produces one word, not forty. A genuine repeat
        # is allowed once the block window has passed.
        recent_repeat = (
            label == self._last_word
            and now - self._last_word_at < self.cfg.repeat_block_seconds
        )
        if not recent_repeat:
            self.words.append(label)
            self._last_word, self._last_word_at = label, now

    def _emit(self):
        phrase = self.words
        self.reset()
        return phrase

    def flush(self):
        """Force out whatever has accumulated. For a 'finish sentence' button."""
        return self._emit() if self.words else None
