"""Metrics for the gloss-to-text stage.

Four families, in descending order of how defensible they are:

1. FAITHFULNESS (automatic, no references needed)
   Did the model add content it was not given, and did it keep the content it
   was given? These are the two failure modes that matter for this system,
   because a wrong sentence here misrepresents what a deaf signer said.

2. SIMILARITY TO A HUMAN REFERENCE (automatic, needs references)
   BLEU and chrF against sentences a human wrote for the same glosses. This is
   what the sign language translation literature reports, so it makes the work
   comparable. chrF is the more trustworthy of the two on short sentences.

3. OPERATIONAL (automatic)
   Latency per call, fallback rate, and output stability across repeated runs.

4. HUMAN RATING (manual)
   The 1-5 scale in the project proposal. Nothing automatic replaces it.

Deliberately NOT included: an LLM scoring its own output. Judging Gemini with
Gemini measures agreement with itself, not quality. See `judge_prompt()` below
if you want it as a supplement, with the bias stated.
"""

import re
import statistics
from collections import Counter


# Words the model may supply. Kept in sync with storyteller.py by the test
# below, so the two cannot drift apart silently.
CONNECTIVES = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did", "my", "your", "his", "her",
    "its", "our", "their", "of", "to", "in", "on", "at", "with", "and", "but",
    "or", "for", "from", "by", "as", "that", "this",
    "s", "not", "no", "very", "so", "then", "up", "down", "out",
}


def tokens(text):
    return re.findall(r"[a-z]+", text.lower())


def content_words(sentence):
    return [t for t in tokens(sentence) if t not in CONNECTIVES]


def gloss_tokens(glosses):
    """Every word a gloss licenses. 'store_or_shop' licenses store, or, shop."""
    out = set()
    for g in glosses:
        for part in re.split(r"[_\-()\s]+", g.lower()):
            if part:
                out.add(part)
                out.add(part.rstrip("s"))
                out.add(part + "s")
    return out


# A closed list of light and motion verbs the model is permitted to supply.
#
# This list exists because INCLUDE has no verb category at all: every one of
# the 50 signs is a noun, adjective, pronoun or time word. A gloss run like
# GOOD_MORNING TEACHER cannot become a sentence unless the model supplies a
# verb. So supplying one is required behaviour, not a failure, and the metric
# must separate it from the real failure, which is inventing an entity.
#
# Keep this list closed and documented. Anything outside it that the glosses
# did not license counts as invented.
SUPPLIABLE_VERBS = {
    "go", "goes", "went", "gone", "going",
    "come", "comes", "came", "coming",
    "see", "sees", "saw", "seen", "seeing",
    "meet", "meets", "met", "meeting",
    "get", "gets", "got", "getting",
    "make", "makes", "made", "making",
    "take", "takes", "took", "taken", "taking",
    "give", "gives", "gave", "given", "giving",
    "say", "says", "said", "saying",
    "want", "wants", "wanted", "wanting",
    "like", "likes", "liked", "liking",
    "feel", "feels", "felt", "feeling",
    "live", "lives", "lived", "living",
    "stay", "stays", "stayed", "staying",
    "work", "works", "worked", "working",
    "visit", "visits", "visited", "visiting",
    "wear", "wears", "wore", "worn", "wearing",
    "own", "owns", "owned", "owning",
    "call", "calls", "called", "calling",
    "look", "looks", "looked", "looking",
    "seem", "seems", "seemed", "seeming",
    "there", "here",     # existential and deictic, grammatical in this role
}


def split_invented(words):
    """Separate a supplied verb from an invented entity.

    Only the second is a real failure. A verb fills a hole the dataset cannot;
    an entity asserts a fact the signer never signed.
    """
    verbs = [w for w in words if w in SUPPLIABLE_VERBS]
    entities = [w for w in words if w not in SUPPLIABLE_VERBS]
    return verbs, entities


# ---- 1. faithfulness ------------------------------------------------------

def invented_words(glosses, sentence):
    """Content words in the sentence that no gloss licenses.

    This is the hallucination measure. Each one is a fact the system asserted
    on behalf of a signer who never signed it.
    """
    allowed = gloss_tokens(glosses)
    return [w for w in content_words(sentence)
            if w not in allowed and w.rstrip("s") not in allowed]


def dropped_glosses(glosses, sentence):
    """Glosses whose meaning does not appear in the sentence at all.

    The opposite failure to hallucination: the signer said something and the
    system silently discarded it. A multi-word gloss counts as kept if any of
    its parts appears, since 'store_or_shop' may legitimately surface as just
    'shop'.
    """
    present = set(tokens(sentence))
    present |= {t.rstrip("s") for t in present}
    present |= {t + "s" for t in present}

    missing = []
    for g in glosses:
        parts = [p for p in re.split(r"[_\-()\s]+", g.lower()) if p]
        if not any(p in present or p.rstrip("s") in present for p in parts):
            missing.append(g)
    return missing


def faithfulness(glosses, sentence):
    """Both directions at once, with supplied verbs separated from inventions.

    The headline number for the report is `n_invented_entities`. A run where
    that is zero is the claim the system makes: the model added grammar, not
    information.
    """
    added = invented_words(glosses, sentence)
    verbs, entities = split_invented(added)
    dropped = dropped_glosses(glosses, sentence)
    return {
        "supplied_verbs": verbs,
        "invented_entities": entities,
        "dropped": dropped,
        "n_supplied_verbs": len(verbs),
        "n_invented_entities": len(entities),
        "n_dropped": len(dropped),
        "coverage": (len(glosses) - len(dropped)) / len(glosses) if glosses else 0.0,
        "faithful": not entities and not dropped,
    }


def paragraph_faithfulness(phrases, paragraph):
    """Faithfulness for a whole paragraph, with cohesion allowed.

    Same rule as the per-sentence check, with one addition: anaphoric
    pronouns and connectives of sequence ("he", "there", "then") are grammar,
    not new entities, because they point back at something already signed.
    They are counted and reported separately so the allowance stays visible.

    `n_invented_entities` remains the number that matters. Zero means the
    paragraph added cohesion and grammar but no information.
    """
    from storyteller import ANAPHORA

    all_glosses = [g for p in phrases for g in p]
    added = invented_words(all_glosses, paragraph)

    cohesion = [w for w in added if w in ANAPHORA]
    rest = [w for w in added if w not in ANAPHORA]
    verbs, entities = split_invented(rest)

    dropped = [p for p in phrases
               if dropped_glosses(p, paragraph) == list(p)]

    return {
        "cohesion_words": cohesion,
        "supplied_verbs": verbs,
        "invented_entities": entities,
        "n_cohesion": len(cohesion),
        "n_supplied_verbs": len(verbs),
        "n_invented_entities": len(entities),
        "phrases_entirely_dropped": dropped,
        "faithful": not entities and not dropped,
        "sentences": len([x for x in re.split(r"[.!?]+", paragraph) if x.strip()]),
        "words": len(tokens(paragraph)),
    }


# ---- 2. similarity to human references ------------------------------------

def corpus_scores(hypotheses, references):
    """BLEU and chrF over the whole set.

    sacrebleu is imported here rather than at module level so the
    faithfulness metrics, which need nothing beyond the standard library,
    keep working without it. Only this function needs the dependency.

    references: list of lists, so several acceptable phrasings per item.
    Corpus-level, not an average of sentence scores, which is what the
    translation literature reports and is less unstable on short text.
    """
    import sacrebleu

    if not hypotheses:
        return {"bleu": 0.0, "chrf": 0.0, "n": 0}

    max_refs = max(len(r) for r in references)
    # sacrebleu wants reference streams, padded with the first reference
    streams = [[(refs[i] if i < len(refs) else refs[0]) for refs in references]
               for i in range(max_refs)]

    bleu = sacrebleu.corpus_bleu(hypotheses, streams)
    chrf = sacrebleu.corpus_chrf(hypotheses, streams)
    return {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "n": len(hypotheses),
        "bleu_detail": bleu.format(),
    }


def exact_match(hypotheses, references):
    """Fraction matching a reference exactly, ignoring case and punctuation."""
    def norm(s):
        return " ".join(tokens(s))
    hits = sum(1 for h, refs in zip(hypotheses, references)
               if norm(h) in {norm(r) for r in refs})
    return hits / len(hypotheses) if hypotheses else 0.0


# ---- 3. operational -------------------------------------------------------

def latency_stats(times_ms):
    """Per-call latency. The proposal asks for response time; this is it."""
    if not times_ms:
        return {}
    s = sorted(times_ms)
    return {
        "mean_ms": round(statistics.mean(s), 1),
        "median_ms": round(statistics.median(s), 1),
        "p90_ms": round(s[int(len(s) * 0.9)] if len(s) > 1 else s[0], 1),
        "max_ms": round(max(s), 1),
        "n": len(s),
    }


def stability(outputs_per_input):
    """How often repeated calls on the same glosses give the same sentence.

    Matters because the story ratings have to be reproducible. At a fixed low
    temperature this should be close to 1.0; if it is not, say so in the report
    rather than quoting a single run as if it were the result.
    """
    rates = []
    for outs in outputs_per_input:
        if len(outs) < 2:
            continue
        normed = [" ".join(tokens(o)) for o in outs]
        most_common = Counter(normed).most_common(1)[0][1]
        rates.append(most_common / len(normed))
    return {
        "mean_agreement": round(statistics.mean(rates), 3) if rates else None,
        "fully_stable": sum(1 for r in rates if r == 1.0),
        "n_inputs": len(rates),
    }


# ---- 4. human rating ------------------------------------------------------

RATING_CRITERIA = [
    ("meaning", "Does the sentence say what the glosses said? 1 = wrong "
                "meaning, 5 = exactly right"),
    ("grammar", "Is it grammatical English? 1 = not a sentence, 5 = fluent"),
    ("context", "Does it fit the story so far? 1 = contradicts it, "
                "5 = follows naturally"),
    ("no_extra", "Did it avoid adding information? 1 = invents events, "
                 "5 = adds nothing"),
]


def rating_sheet(records):
    """A blind rating sheet: glosses and sentence only, no model name.

    Shuffle before handing to raters, and get at least two raters so you can
    report agreement rather than one person's opinion.
    """
    lines = ["# Story rating sheet",
             "",
             "Rater: ______________________   Date: ____________",
             "",
             "Fill one sheet per rater, save as rating_sheet_<name>.md, then:",
             "    python src/score_ratings.py checkpoints/rating_sheet_*.md",
             "",
             "Rate each 1 to 5. Criteria:", ""]
    for key, desc in RATING_CRITERIA:
        lines.append(f"- **{key}**: {desc}")
    lines += ["", "---", ""]
    for i, r in enumerate(records, 1):
        lines.append(f"## {i}")
        lines.append(f"Signs: `{' '.join(r['glosses']).upper()}`")
        if r.get("context"):
            lines.append(f"Story so far: {r['context']}")
        lines.append(f"Output: **{r['sentence']}**")
        lines.append("")
        lines.append("| " + " | ".join(k for k, _ in RATING_CRITERIA) + " |")
        lines.append("|" + "---|" * len(RATING_CRITERIA))
        lines.append("|" + "   |" * len(RATING_CRITERIA))
        lines.append("")
    return "\n".join(lines)


def judge_prompt(glosses, sentence, context=""):
    """Optional LLM-as-judge prompt.

    Use a DIFFERENT model from the one that generated the text. Scoring Gemini
    output with Gemini measures self-agreement, not quality, and a reviewer
    will say so. Report it as a supplement to human ratings, never instead.
    """
    return (
        "You are grading a sign-language gloss-to-English system.\n"
        f"{'Story so far: ' + context if context else ''}\n"
        f"Glosses: {' '.join(g.upper() for g in glosses)}\n"
        f"Output: {sentence}\n\n"
        "Score 1-5 on each: meaning, grammar, context, no_extra.\n"
        "Reply as JSON only: {\"meaning\":n,\"grammar\":n,"
        "\"context\":n,\"no_extra\":n}"
    )
