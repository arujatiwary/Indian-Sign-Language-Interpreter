"""Evaluate the gloss-to-text stage and write a metrics file.

    export GEMINI_API_KEY=...
    python src/eval_llm.py                      # one pass
    python src/eval_llm.py --repeats 3          # also measure output stability
    python src/eval_llm.py --temperature 0.0    # compare settings

Writes checkpoints/llm_eval.json and checkpoints/rating_sheet.md.

The run refuses to report quality scores if the API failed, because the
offline template produces output that looks like a bad model rather than a
dead connection, and those two must never be confused in a report.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm_eval import (  # noqa: E402
    corpus_scores, exact_match, faithfulness, latency_stats, rating_sheet,
    stability,
)
from storyteller import Storyteller  # noqa: E402


def load_refs(path):
    data = json.loads(Path(path).read_text())
    return data["single"], data["stories"], data.get("provenance", "draft")


def run_once(st, glosses, delay):
    """One call, timed. Returns sentence, latency, error."""
    time.sleep(delay)
    t0 = time.perf_counter()
    sentence = st.add(glosses)
    ms = (time.perf_counter() - t0) * 1000
    return sentence, ms, st.last_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="data/references.json")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--repeats", type=int, default=1,
                    help="calls per input; >1 measures output stability")
    ap.add_argument("--delay", type=float, default=6.0,
                    help="seconds between calls, to stay inside a free-tier "
                         "rate limit")
    ap.add_argument("--out", default="checkpoints/llm_eval.json")
    args = ap.parse_args()

    singles, stories, provenance = load_refs(args.refs)
    # A reference you did not write yourself is not a human reference. These
    # drafts were generated alongside the system, so scoring against them
    # measures the model against its own author's guess, and BLEU near 99 says
    # the two agree, not that the output is good. The scores stay computed,
    # because they are a useful regression check, but they are not reported
    # until the file declares that a person wrote them.
    refs_are_human = str(provenance).lower().startswith("human")
    st = Storyteller(temperature=args.temperature)

    if st.provider == "none":
        raise SystemExit(
            "no usable API key, so every call would fall back to the offline "
            "template and the scores would describe the template.\n"
            "Put your key in .env (see .env.example) or export it:\n"
            "    export GEMINI_API_KEY=...\n"
            "Free key: https://aistudio.google.com/apikey"
        )
    print(f"provider {st.provider} | model {st.model or 'default'} | "
          f"temperature {args.temperature} | repeats {args.repeats}\n")

    records, latencies, failures = [], [], 0
    repeats_by_input = []

    print("single phrases")
    for item in singles:
        glosses, refs = item["glosses"], item["refs"]
        outs = []
        for _ in range(args.repeats):
            st.clear()
            sentence, ms, err = run_once(st, glosses, args.delay)
            outs.append(sentence)
            latencies.append(ms)
            failures += err is not None
        repeats_by_input.append(outs)

        sentence = outs[0]
        f = faithfulness(glosses, sentence)
        records.append({"glosses": glosses, "sentence": sentence, "refs": refs,
                        "context": "", **f, "api_error": st.last_error})
        mark = "" if f["faithful"] else \
            f"   <-- invented {f['invented_entities']} dropped {f['dropped']}"
        print(f"  {' '.join(glosses):34} {sentence}{mark}")

    print("\nstories")
    for story in stories:
        st.clear()
        for turn in story["turns"]:
            glosses, refs = turn["glosses"], turn["refs"]
            context = st.story
            sentence, ms, err = run_once(st, glosses, args.delay)
            latencies.append(ms)
            failures += err is not None
            f = faithfulness(glosses, sentence)
            records.append({"glosses": glosses, "sentence": sentence,
                            "refs": refs, "context": context, "in_story": True,
                            **f, "api_error": st.last_error})
            mark = "" if f["faithful"] else \
                f"   <-- invented {f['invented_entities']} dropped {f['dropped']}"
            print(f"  {' '.join(glosses):34} {sentence}{mark}")
        print(f"  [{story['name']}] {st.story}\n")

    # ---- aggregate --------------------------------------------------
    n = len(records)
    hyps = [r["sentence"] for r in records]
    refs = [r["refs"] for r in records]

    clean = sum(1 for r in records if r["faithful"])
    invented = sum(r["n_invented_entities"] for r in records)
    dropped = sum(r["n_dropped"] for r in records)
    verbs = sum(r["n_supplied_verbs"] for r in records)
    coverage = sum(r["coverage"] for r in records) / n

    results = {
        "config": {"provider": st.provider,
                   "model_requested": st.model,
                   "model_served": st.model_used,
                   "temperature": args.temperature,
                   "repeats": args.repeats,
                   "references": provenance,
                   "similarity_reportable": refs_are_human},
        "faithfulness": {
            "phrases": n,
            "fully_faithful": clean,
            "fully_faithful_rate": round(clean / n, 3),
            "invented_entities": invented,
            "invented_per_phrase": round(invented / n, 3),
            "dropped_glosses": dropped,
            "gloss_coverage": round(coverage, 3),
            "supplied_verbs": verbs,
        },
        "similarity": corpus_scores(hyps, refs),
        "exact_match": round(exact_match(hyps, refs), 3),
        "latency": latency_stats(latencies),
        "reliability": {"api_failures": failures, "calls": len(latencies),
                        "fallback_rate": round(failures / len(latencies), 3)},
        "records": records,
    }
    if args.repeats > 1:
        results["stability"] = stability(repeats_by_input)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    sheet = out.parent / "rating_sheet.md"
    sheet.write_text(rating_sheet(records))

    # ---- report -----------------------------------------------------
    print("=" * 56)
    if failures:
        print(f"WARNING: {failures}/{len(latencies)} calls failed and used the "
              "offline template.")
        print("The scores below describe the template, not the model. Fix the "
              "API before reporting them.\n")

    fa = results["faithfulness"]
    print(f"fully faithful       {fa['fully_faithful']}/{n} "
          f"({fa['fully_faithful_rate']:.1%})")
    print(f"invented entities    {fa['invented_entities']} "
          f"({fa['invented_per_phrase']} per phrase)")
    print(f"gloss coverage       {fa['gloss_coverage']:.1%}")
    print(f"dropped glosses      {fa['dropped_glosses']}")
    print(f"verbs supplied       {fa['supplied_verbs']} (expected: INCLUDE "
          "has no verbs)")
    if refs_are_human:
        print(f"BLEU / chrF          {results['similarity']['bleu']} / "
              f"{results['similarity']['chrf']}")
        print(f"exact match          {results['exact_match']:.1%}")
    else:
        print(f"BLEU / chrF          NOT REPORTABLE "
              f"(computed: {results['similarity']['bleu']} / "
              f"{results['similarity']['chrf']})")
        print("  data/references.json is marked "
              f"\"{provenance}\". These references were drafted alongside the")
        print("  system, so the score measures agreement with its author, not "
              "translation")
        print("  quality. Rewrite them yourself, then set "
              "\"provenance\": \"human\" in that file.")
    print(f"model served         {st.model_used or 'unknown'}"
          + ("" if st.model_used == st.model or st.model is None
             else f"  (requested {st.model})"))
    if results["latency"]:
        print(f"latency              {results['latency']['median_ms']:.0f} ms "
              f"median, {results['latency']['p90_ms']:.0f} ms p90")
    if args.repeats > 1:
        print(f"output stability     {results['stability']['mean_agreement']}")
    print(f"\nwrote {out}")
    print(f"wrote {sheet}  (blind sheet for human 1-5 ratings)")


if __name__ == "__main__":
    main()
