"""
analysis/error_analysis.py — Where does the system fail, and what does healing change?

    python -m analysis.error_analysis results/hotpotqa_n200_s42/self_healing.jsonl

Categories (short-answer QA):
  correct          EM = 1
  partial          EM = 0 but token F1 > 0 (right idea, wrong span/format)
  retrieval_miss   a gold supporting passage never reached the generator
  reasoning_error  all gold passages were in context, answer still wrong

Healing transitions compare the initial answer with the final one:
  fixed (wrong -> right), broken (right -> wrong), kept_right, kept_wrong.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any, Dict, List


def categorize(rec: Dict[str, Any]) -> str:
    if rec.get("em", 0) >= 1:
        return "correct"
    if rec.get("f1", 0) > 0:
        return "partial"
    if rec.get("support_recall", 1.0) < 1.0:
        return "retrieval_miss"
    return "reasoning_error"


def transition(rec: Dict[str, Any]) -> str:
    before, after = rec.get("initial_em", 0) >= 1, rec.get("em", 0) >= 1
    if before and after:
        return "kept_right"
    if not before and after:
        return "fixed"
    if before and not after:
        return "broken"
    return "kept_wrong"


def analyze(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in records if not r.get("error")]
    n = max(1, len(ok))
    cats = Counter(categorize(r) for r in ok)
    trans = Counter(transition(r) for r in ok)
    by_failure: Dict[str, Counter] = defaultdict(Counter)
    by_strategy: Dict[str, Counter] = defaultdict(Counter)
    for r in ok:
        if r.get("failure_detected"):
            by_failure[r.get("failure_type") or "?"][transition(r)] += 1
            rounds = r.get("rounds", [])
            sel = r.get("selected_round", 0)
            strategy = rounds[sel]["produced_by"] if sel < len(rounds) else "initial"
            by_strategy[strategy][transition(r)] += 1
    return {
        "n": len(ok),
        "categories": {k: {"count": v, "pct": v / n} for k, v in cats.most_common()},
        "transitions": dict(trans),
        "by_failure_type": {k: dict(v) for k, v in by_failure.items()},
        "by_selected_strategy": {k: dict(v) for k, v in by_strategy.items()},
    }


def case_studies(records: List[Dict[str, Any]], kind: str = "fixed", n: int = 3) -> str:
    """Markdown write-ups of a few healing successes (kind='fixed') or failures ('broken')."""
    picks = [r for r in records if not r.get("error") and transition(r) == kind][:n]
    out = []
    for r in picks:
        out.append(f"#### {r['question']}\n")
        out.append(f"- Gold: **{r['gold']}**")
        for rd in r.get("rounds", []):
            verdict = "" if rd.get("has_failure") is None else (
                f" — monitor: {'FAIL ' + rd.get('failure_type', '') if rd['has_failure'] else 'PASS'}"
                f" (tier {rd.get('monitor_tier')}, conf {rd.get('confidence')})")
            out.append(f"- Round {rd['round']} [{rd['produced_by']}]: `{rd['answer']}`{verdict}")
        out.append(f"- Final: `{r['answer']}`\n")
    return "\n".join(out) if out else f"_No '{kind}' cases._"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="per-sample results file, e.g. results/<tag>/self_healing.jsonl")
    ap.add_argument("--cases", type=int, default=3)
    args = ap.parse_args()
    with open(args.jsonl, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    print(json.dumps(analyze(records), indent=2))
    print("\n### Healing successes\n")
    print(case_studies(records, "fixed", args.cases))
    print("\n### Healing regressions\n")
    print(case_studies(records, "broken", args.cases))


if __name__ == "__main__":
    main()
