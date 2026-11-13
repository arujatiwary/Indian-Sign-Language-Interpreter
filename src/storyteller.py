"""Turn recognised signs into English sentences and a running story.

The recognised signs are glosses: word labels in signing order, without English
grammar. INCLUDE has no verb category, so a gloss run like

    GOOD_MORNING TEACHER

cannot become a sentence without the model supplying a verb. The rule enforced
here, in the prompt and in the checker below:

    The model may add grammatical connectives (is, was, the, my, with, and).
    It may not add entities, places, times, or events that were not signed.

"Good morning, teacher." is fine. "Good morning, teacher, I saw your dog at the
market." is not, because dog and market were never signed.

No model is trained here. A frozen hosted model is prompted zero-shot, the same
relationship the vision half has with MediaPipe.
"""

import json
import os
from pathlib import Path
import re
import ssl
import time
import urllib.error
import urllib.request

SYSTEM_PROMPT = """You convert Indian Sign Language glosses into English.

Glosses are word labels in signing order. They have no grammar, no tense
markers, no articles, and no verbs. Your job is to produce the plainest English
sentence those words support.

Rules:
1. Every content word in your sentence must come from the glosses. Do not
   introduce any person, animal, object, place, colour, or time that was not
   signed.
2. You may add only grammatical connectives: forms of "be" and "have",
   articles, prepositions, possessives, and conjunctions.
3. Do not invent events. If the glosses do not describe an action, do not
   supply one.
4. Keep it to one short sentence. Do not embellish.
5. If a previous story is supplied, keep the same subject and tense where the
   glosses allow it. Do not restate or summarise the earlier sentences.
6. Output only the sentence. No preamble, no quotes, no explanation.

Examples:
Glosses: GOOD_MORNING TEACHER
Sentence: Good morning, teacher.

Glosses: I HAPPY
Sentence: I am happy.

Glosses: BROTHER DOG BLACK
Sentence: My brother's dog is black.

Glosses: MONDAY I HOUSE
Sentence: On Monday I was at the house."""

NARRATE_PROMPT = """You turn a sequence of Indian Sign Language glosses into
one short, connected paragraph of English.

The glosses are everything the signer said, in order, grouped into phrases.
They have no grammar, no tense, no articles and no verbs, because the sign
vocabulary contains none.

Write the paragraph the way a person would retell what was signed.

Rules:
1. Every PERSON, ANIMAL, OBJECT, PLACE, COLOUR and TIME in your paragraph must
   come from the glosses. Introduce nothing new. If the signer did not sign it,
   it does not appear.
2. You may supply what English grammar requires: verbs of being and simple
   motion, articles, prepositions, possessives, conjunctions.
3. CONNECT the sentences. This is the part that matters. Refer back to things
   already mentioned with pronouns (he, she, it, they, there, that), and join
   clauses with words like and, then, so, while, where. A pronoun pointing at
   something already signed is grammar, not new information.
4. Three to five sentences. Keep it plain. No adjectives that were not signed,
   no mood, no scene-setting, no invented detail.
5. Treat the phrase grouping as sentence hints, not hard boundaries. Merge two
   short phrases into one sentence when that reads better.
6. Output only the paragraph. No preamble, no heading, no quotes.

Example:
Phrases: MONDAY | I HOUSE | BROTHER FATHER | COW WINDOW | LOUD | I HAPPY
Paragraph: On Monday I was at the house. My brother and my father were there
too. There was a cow at the window, and it was loud. I was happy.

Example:
Phrases: GOOD_MORNING TEACHER | I HAPPY | THANK_YOU TEACHER
Paragraph: Good morning, teacher. I am happy to be here, and I thank you."""


# Pronouns and deictics that point back at something already signed. In
# paragraph mode these are cohesion, not new entities: "the dog ... it was
# big" introduces nothing. They stay OUT of CONNECTIVES because in
# single-sentence mode there is nothing for them to refer back to, and because
# "it" and "you" are themselves signs in the vocabulary.
ANAPHORA = {
    # third person, pointing back at a signed entity
    "he", "him", "his", "she", "her", "hers", "they", "them", "theirs",
    # first person plural: "I" plus an already-signed companion
    "we", "us", "our", "ours",
    # "it" and deictics. "it" is also a sign in the vocabulary, but inside a
    # paragraph it is almost always anaphoric ("a cow ... it was loud").
    # Allowing every pronoun is safe for the metric that matters: a pronoun is
    # never itself an entity, so an invented referent is still caught by its
    # noun. "We bought bread" passes "we" and fails on "bread".
    "it", "its", "there", "that", "those", "these", "who", "which",
    # sequence and linking
    "too", "also", "both", "while", "where", "when", "then", "after",
    "before", "because", "since", "still", "again",
}


# Words the model may add freely. Everything else in its output must trace back
# to a gloss, or the constraint has been broken.
CONNECTIVES = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did", "my", "your", "his", "her",
    "its", "our", "their", "of", "to", "in", "on", "at", "with", "and", "but",
    "or", "for", "from", "by", "as", "that", "this",
    "s", "not", "no", "very", "so", "then", "up", "down", "out",
}
# Deliberately absent: "i", "it", "you", "there", "here". Each is either a sign
# in the vocabulary or a deictic that could smuggle in an unsigned referent, so
# each must be licensed by a gloss like any other content word.


class Storyteller:
    """Holds the running story and turns gloss runs into sentences."""

    def __init__(self, provider=None, model=None, temperature=0.2, max_context=4):
        self.provider = provider or _detect_provider()
        self.model = model
        self.temperature = temperature
        self.max_context = max_context   # sentences of story sent as context
        self.sentences = []
        self.last_error = None
        self.model_used = None     # the model that answered, not the one asked

    # ---- public API ---------------------------------------------------

    @property
    def story(self):
        return " ".join(self.sentences)

    def add(self, glosses):
        """Take a finished phrase, return the sentence added to the story."""
        if not glosses:
            return None
        try:
            sentence = self._generate(glosses)
            self.last_error = None
        except Exception as e:            # network, auth, rate limit, anything
            self.last_error = repr(e)
            sentence = fallback_sentence(glosses)
        self.sentences.append(sentence)
        return sentence

    def clear(self):
        self.sentences = []

    def narrate(self, phrases):
        """Turn every phrase signed so far into one connected paragraph.

        This is the mode the project proposal actually describes. Calling
        add() per phrase gives correct but disconnected sentences, because
        each call sees one phrase and is told to produce one sentence. A
        paragraph needs the whole sequence at once so the model can carry a
        subject across sentences and refer back to it.

        phrases: list of gloss lists, e.g. [["monday"], ["i","house"]]
        """
        phrases = [p for p in phrases if p]
        if not phrases:
            return None

        joined = " | ".join(" ".join(g.upper() for g in p) for p in phrases)
        user = f"Phrases: {joined}\nParagraph:"

        try:
            text = _call_api(self.provider, self.model, NARRATE_PROMPT, user,
                             self.temperature)
            self.last_error = None
            self.model_used = last_served_model()
            return " ".join(_clean_paragraph(text).split())
        except Exception as e:
            self.last_error = repr(e)
            return fallback_paragraph(phrases)

    # ---- generation ---------------------------------------------------

    def _generate(self, glosses):
        if self.provider == "none":
            raise RuntimeError("no API key configured")

        context = " ".join(self.sentences[-self.max_context:])
        gloss_text = " ".join(g.upper() for g in glosses)
        user = (
            (f"Story so far: {context}\n\n" if context else "")
            + f"Glosses: {gloss_text}\nSentence:"
        )

        text = _call_api(self.provider, self.model, SYSTEM_PROMPT, user,
                         self.temperature)
        self.model_used = last_served_model()
        return _clean(text)


# ---- the constraint check, also used as a test and as a reported metric ----

def content_words(sentence):
    """Words in the sentence that are not grammatical connectives."""
    tokens = re.findall(r"[a-z]+", sentence.lower())
    return [t for t in tokens if t not in CONNECTIVES]


def gloss_tokens(glosses):
    """Every word a gloss licenses. 'store_or_shop' licenses store, or, shop."""
    out = set()
    for g in glosses:
        for part in re.split(r"[_\-()\s]+", g.lower()):
            if part:
                out.add(part)
                out.add(part.rstrip("s"))       # dog / dogs
                out.add(part + "s")
    return out


def invented_words(glosses, sentence):
    """Content words in the sentence that no gloss licenses.

    This is the number to report. Zero across a test set is the claim that the
    language model added grammar but not information.
    """
    allowed = gloss_tokens(glosses)
    out = []
    for w in content_words(sentence):
        if w in allowed or w.rstrip("s") in allowed:
            continue
        out.append(w)
    return out


# ---- offline fallback -----------------------------------------------------

def fallback_paragraph(phrases):
    """Offline paragraph. Plain, and visibly not model output."""
    return " ".join(fallback_sentence(p) for p in phrases if p)


def _clean_paragraph(text):
    """Strip wrappers but keep the sentence structure, unlike _clean()."""
    text = text.strip().strip('"').strip()
    text = re.sub(r"^(paragraph|output|story)\s*:\s*", "", text, flags=re.I)
    return text.strip()


def fallback_sentence(glosses):
    """Template output when the API is unreachable.

    Deliberately plain. It keeps a live demo alive without pretending to be
    the model's work.
    """
    words = [g.replace("_", " ") for g in glosses]
    if not words:
        return ""
    text = ", ".join(words)
    return text[0].upper() + text[1:] + "."


# ---- provider adapters ----------------------------------------------------

def _load_dotenv():
    """Read .env from the project root into the environment.

    Without this the key has to be exported in every new terminal, and the
    failure is silent: a missing key means every call falls back to the
    offline template, which produces output that looks like a bad model
    rather than a missing key. That cost a whole evaluation run.

    Existing environment variables win, so an explicit export still overrides
    the file.
    """
    for base in (Path(__file__).resolve().parent.parent, Path.cwd()):
        env = base / ".env"
        if not env.exists():
            continue
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
        return True
    return False


_load_dotenv()


# A key left as the .env placeholder is worse than no key: the provider looks
# configured, every call 400s, and the output falls back to the template, which
# reads as a bad model rather than a setup mistake.
PLACEHOLDERS = {"", "your_key_here", "paste_your_key_here", "none", "xxx",
                "changeme", "..."}


def _usable(name):
    val = (os.environ.get(name) or "").strip()
    return val if val.lower() not in PLACEHOLDERS else ""


def _detect_provider():
    if _usable("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _usable("OPENAI_API_KEY"):
        return "openai"
    if _usable("GEMINI_API_KEY"):
        return "gemini"
    return "none"


# Tried in order when the chosen model is overloaded. A 503 is the server
# saying "too busy", not a problem with the request, so the same call usually
# succeeds on a less loaded model. Without this, one busy model kills a live
# demo even though the key, the prompt and the network are all fine.
# Ordered fastest first. A measured run showed 18 s median latency with the
# larger flash models under load; this system generates a sentence per signed
# phrase, so speed matters more here than the small quality difference on a
# task that is mostly inserting articles and copulas.
GEMINI_FALLBACKS = [
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
]

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.1-flash-lite",
}


def _ssl_context():
    """Build a context with a real CA bundle.

    python.org builds on macOS do not use the system keychain, so urllib has no
    trusted roots until someone runs Install Certificates.command. certifi ships
    the bundle, so use it when present and fall back to the default otherwise.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL = _ssl_context()


RETRY_CODES = {429, 500, 502, 503, 504}


def _post(url, payload, headers, timeout=60, retries=4):
    """POST with backoff on rate limits and transient server errors.

    Free API tiers cap requests per minute, and a batch evaluation trips that
    immediately. Retrying is the difference between 5 of 24 calls landing and
    all of them.
    """
    body = json.dumps(payload).encode()
    delay = 2.0
    last = None

    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers,
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_CODES:
                detail = e.read().decode()[:300]
                raise RuntimeError(f"HTTP {e.code}: {detail}") from None
            last = e
        except urllib.error.URLError as e:
            last = e
        except (TimeoutError, OSError) as e:
            # A socket timeout is not an HTTPError or a URLError, so without
            # this it escaped the retry loop entirely, skipped the model
            # fallback, and went straight to the offline template. Gemini 3.x
            # spends time on internal reasoning before answering, so a slow
            # reply is normal, not a failure.
            last = e
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"gave up after {retries} attempts: {last!r}")


# The model that actually served the most recent call. The requested model and
# the answering model are not the same thing: Gemini falls down a chain when a
# model is retired or overloaded, and "model: null" in a results file records
# what was asked for, which is not what the numbers describe.
_LAST_SERVED = None


def last_served_model():
    """The model that answered most recently, or None before any call."""
    return _LAST_SERVED


def _call_api(provider, model, system, user, temperature):
    global _LAST_SERVED
    model = model or DEFAULT_MODELS[provider]
    _LAST_SERVED = model          # overwritten below if a fallback answers

    if provider == "anthropic":
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {"model": model, "max_tokens": 200, "temperature": temperature,
             "system": system, "messages": [{"role": "user", "content": user}]},
            {"content-type": "application/json",
             "x-api-key": os.environ["ANTHROPIC_API_KEY"],
             "anthropic-version": "2023-06-01"},
        )
        return data["content"][0]["text"]

    if provider == "openai":
        data = _post(
            "https://api.openai.com/v1/chat/completions",
            {"model": model, "temperature": temperature, "max_tokens": 200,
             "messages": [{"role": "system", "content": system},
                          {"role": "user", "content": user}]},
            {"content-type": "application/json",
             "authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        )
        return data["choices"][0]["message"]["content"]

    if provider == "gemini":
        return _gemini_with_fallback(model, system, user, temperature)

    raise ValueError(f"unknown provider {provider!r}")


# The model that last answered. Once a model is known to be up, going
# straight to it saves walking past the dead ones on every single call.
_LAST_GOOD = None

RETRYABLE = ("503", "429", "overload", "404", "UNAVAILABLE",
             "timed out", "timeout", "TimeoutError")


def _set_served(model):
    global _LAST_SERVED
    _LAST_SERVED = model


def _gemini_with_fallback(model, system, user, temperature):
    """Try each model once, then retry the whole chain with backoff.

    The order matters more than it looks. Spending a full exponential backoff
    on a model that is down, before even trying the next one, cost about 30
    seconds per dead model: a measured run sat at 50 s median latency where the
    working model answered in about 2 s. One quick pass across all models finds
    a live one immediately; the slow retry only happens when everything is
    genuinely down.
    """
    global _LAST_GOOD

    chain = [model] + [m for m in GEMINI_FALLBACKS if m != model]
    if _LAST_GOOD and _LAST_GOOD in chain:
        chain = [_LAST_GOOD] + [m for m in chain if m != _LAST_GOOD]

    last = None

    # pass 1: one attempt each, no waiting
    for candidate in chain:
        try:
            out = _gemini_once(candidate, system, user, temperature, retries=1)
            if candidate != _LAST_GOOD:
                print(f"  [using {candidate}]")
                _LAST_GOOD = candidate
            _set_served(candidate)
            return out
        except RuntimeError as e:
            if not any(c in str(e) for c in RETRYABLE):
                raise          # a real error fails identically everywhere
            last = e

    # pass 2: everything looked down, so wait properly before giving up
    print("  [all models busy, retrying with backoff]")
    for candidate in chain:
        try:
            out = _gemini_once(candidate, system, user, temperature, retries=3)
            _LAST_GOOD = candidate
            _set_served(candidate)
            return out
        except RuntimeError as e:
            if not any(c in str(e) for c in RETRYABLE):
                raise
            last = e

    raise RuntimeError(f"all Gemini models unavailable; last: {last}")


def _gemini_once(model, system, user, temperature, retries=4):
    if True:
        key = os.environ["GEMINI_API_KEY"]

        # Gemini 2.5 and later reason before answering, and those thinking
        # tokens come out of maxOutputTokens. With a tight budget the answer is
        # truncated or empty, with finishReason MAX_TOKENS. So: keep thinking
        # minimal and leave generous headroom. The knob differs by generation.
        gen = {"temperature": temperature, "maxOutputTokens": 800}
        if model.startswith("gemini-3") or model.startswith("gemini-4"):
            gen["thinkingConfig"] = {"thinkingLevel": "low"}
        else:
            gen["thinkingConfig"] = {"thinkingBudget": 0}

        data = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
            {"system_instruction": {"parts": [{"text": system}]},
             "contents": [{"parts": [{"text": user}]}],
             "generationConfig": gen},
            {"content-type": "application/json"},
            retries=retries,
        )
        return _gemini_text(data)


def _gemini_text(data):
    """Pull the answer out, and say plainly when there is not one."""
    try:
        cand = data["candidates"][0]
    except (KeyError, IndexError):
        raise RuntimeError(f"no candidate in response: {str(data)[:300]}")

    parts = cand.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    if text:
        return text

    reason = cand.get("finishReason", "unknown")
    if reason == "MAX_TOKENS":
        raise RuntimeError(
            "the token budget was spent on thinking before any answer was "
            "produced; raise maxOutputTokens or lower the thinking level"
        )
    raise RuntimeError(f"empty response, finishReason={reason}")


def _clean(text):
    text = text.strip().strip('"').strip()
    text = re.sub(r"^(sentence|output)\s*:\s*", "", text, flags=re.I)
    return text.split("\n")[0].strip()