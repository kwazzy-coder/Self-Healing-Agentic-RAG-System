"""
demo.py — Watch the self-healing loop on 5 hand-picked questions.

    python demo.py                      # uses LLM_BACKEND from .env / environment
    python demo.py --provider groq      # pick a provider explicitly

With no API key at all it falls back to local Flan-T5 (weak, but runs offline
after a one-time model download).
"""
from __future__ import annotations

import argparse
import logging

DEMO_SAMPLES = [
    {
        "question": "What is the capital of France?",
        "documents": [
            "Title: France\nFrance is a country in Western Europe. Its capital and largest city is Paris.",
            "Title: Paris\nParis is the capital and most populous city of France, situated on the River Seine.",
            "Title: Germany\nGermany is a country in Central Europe. Its capital is Berlin.",
            "Title: Europe\nEurope is a continent with many countries including France, Germany, and Italy.",
        ],
        "gold_answer": "Paris",
        "question_type": None,
        "note": "simple factual — should pass the monitor at Tier 2",
    },
    {
        "question": "Who invented the telephone?",
        "documents": [
            "Title: Radio\nThe radio was invented by Guglielmo Marconi in 1895.",
            "Title: Television\nJohn Logie Baird demonstrated the first television in 1926.",
            "Title: Internet\nThe internet originated from ARPANET, a US Defense project from the 1960s.",
            "Title: Printing Press\nJohannes Gutenberg invented the printing press around 1440.",
        ],
        "gold_answer": "Alexander Graham Bell",
        "question_type": None,
        "note": "answer NOT in the documents — the right behaviour is to flag it, not to guess",
    },
    {
        "question": "What is the capital of the country where the Eiffel Tower is located?",
        "documents": [
            "Title: Eiffel Tower\nThe Eiffel Tower is a wrought-iron lattice tower located in Paris, France. Built in 1889.",
            "Title: France\nFrance is a country in Western Europe. Its capital and largest city is Paris.",
            "Title: Paris\nParis is the capital city of France and is home to many famous landmarks.",
            "Title: London\nLondon is the capital of England and the United Kingdom.",
        ],
        "gold_answer": "Paris",
        "question_type": "bridge",
        "note": "2-hop bridge question — planner decomposes it",
    },
    {
        "question": "What causes COVID-19?",
        "documents": [
            "Title: COVID-19\nCOVID-19 is caused by the SARS-CoV-2 coronavirus, first identified in Wuhan, China in "
            "December 2019.",
            "Title: SARS-CoV-2\nSARS-CoV-2 is a novel betacoronavirus. It spreads primarily through respiratory droplets.",
            "Title: Symptoms\nCommon symptoms of COVID-19 include fever, dry cough, and fatigue.",
            "Title: Vaccination\nCOVID-19 vaccines train the immune system to recognize and fight SARS-CoV-2.",
        ],
        "gold_answer": "SARS-CoV-2 coronavirus",
        "question_type": None,
        "note": "grounded factual",
    },
    {
        "question": "Was the Eiffel Tower built before or after the Statue of Liberty?",
        "documents": [
            "Title: Eiffel Tower\nThe Eiffel Tower was constructed between 1887 and 1889 in Paris, France.",
            "Title: Statue of Liberty\nThe Statue of Liberty was a gift from France to the United States, dedicated "
            "on October 28, 1886.",
            "Title: French Architecture\nFrance has a long tradition of iconic architectural structures built in the "
            "19th century.",
            "Title: American Landmarks\nThe United States has many famous landmarks gifted or built in the 19th century.",
        ],
        "gold_answer": "after",
        "question_type": "comparison",
        "note": "comparison question — parallel decomposition",
    },
]


def run_demo(provider: str | None = None) -> None:
    from agent.llm_reasoner import LLMReasoner
    from evaluation.metrics import exact_match, token_f1
    from pipeline import Components, SelfHealingRAGPipeline

    pipeline = SelfHealingRAGPipeline(components=Components.build(llm=LLMReasoner(provider)))
    llm = pipeline.llm
    print("=" * 70)
    print(f"  Self-Healing Agentic RAG — demo   (LLM: {llm.backend}/{llm.model})")
    print("=" * 70)

    totals = {"em": 0.0, "f1": 0.0, "flagged": 0, "healed": 0, "calls": 0}
    for i, s in enumerate(DEMO_SAMPLES, 1):
        out = pipeline.run(s["question"], s["documents"], question_type=s["question_type"])
        em, f1 = exact_match(out.final_answer, s["gold_answer"]), token_f1(out.final_answer, s["gold_answer"])
        totals["em"] += em
        totals["f1"] += f1
        totals["flagged"] += int(out.failure_detected)
        totals["healed"] += int(out.selected_round > 0)
        totals["calls"] += out.cost_data.get("n_llm_calls", 0)
        mark = "✓" if em else ("~" if f1 > 0.3 else "✗")
        print(f"\n[{i}] {s['question']}\n    ({s['note']})")
        print(f"    gold: {s['gold_answer']} | answer: {out.final_answer} {mark}")
        for step in out.trace:
            print(f"      → {step}")
        print(f"    cost: {out.cost_data.get('n_llm_calls', 0)} LLM calls, "
              f"{out.cost_data.get('total_latency_seconds', 0):.1f}s")

    n = len(DEMO_SAMPLES)
    print("\n" + "=" * 70)
    print(f"  EM {totals['em'] / n:.2f} | F1 {totals['f1'] / n:.2f} | flagged {totals['flagged']}/{n} | "
          f"answers replaced by healing {totals['healed']} | LLM calls {totals['calls']}")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    run_demo(args.provider)
