"""
demo.py — Tests the Self-Healing RAG pipeline on 5 clear samples.
Run: python demo.py
"""
import os

DEMO_SAMPLES = [
    {
        # Sample 1: Simple factual — answer clearly in docs
        "question": "What is the capital of France?",
        "documents": [
            "Title: France\nFrance is a country in Western Europe. Its capital and largest city is Paris.",
            "Title: Paris\nParis is the capital and most populous city of France, situated on the River Seine.",
            "Title: Germany\nGermany is a country in Central Europe. Its capital is Berlin.",
            "Title: Europe\nEurope is a continent with many countries including France, Germany, and Italy.",
        ],
        "gold_answer": "Paris",
        "question_type": None,
    },
    {
        # Sample 2: Answer NOT in docs — should trigger failure + correction
        "question": "Who invented the telephone?",
        "documents": [
            "Title: Radio\nThe radio was invented by Guglielmo Marconi in 1895.",
            "Title: Television\nJohn Logie Baird demonstrated the first television in 1926.",
            "Title: Internet\nThe internet originated from ARPANET, a US Defense project from the 1960s.",
            "Title: Printing Press\nJohannes Gutenberg invented the printing press around 1440.",
        ],
        "gold_answer": "Alexander Graham Bell",
        "question_type": None,
    },
    {
        # Sample 3: Multi-hop bridge — needs 2 hops (Eiffel → France → Paris)
        "question": "What is the capital of the country where the Eiffel Tower is located?",
        "documents": [
            "Title: Eiffel Tower\nThe Eiffel Tower is a wrought-iron lattice tower located in Paris, France. Built in 1889.",
            "Title: France\nFrance is a country in Western Europe. Its capital and largest city is Paris.",
            "Title: Paris\nParis is the capital city of France and is home to many famous landmarks.",
            "Title: London\nLondon is the capital of England and the United Kingdom.",
        ],
        "gold_answer": "Paris",
        "question_type": "bridge",
    },
    {
        # Sample 4: RAGBench-style grounded QA — should pass cleanly
        "question": "What causes COVID-19?",
        "documents": [
            "Title: COVID-19\nCOVID-19 is caused by the SARS-CoV-2 coronavirus, first identified in Wuhan, China in December 2019.",
            "Title: SARS-CoV-2\nSARS-CoV-2 is a novel betacoronavirus. It spreads primarily through respiratory droplets.",
            "Title: Symptoms\nCommon symptoms of COVID-19 include fever, dry cough, and fatigue.",
            "Title: Vaccination\nCOVID-19 vaccines train the immune system to recognize and fight SARS-CoV-2.",
        ],
        "gold_answer": "SARS-CoV-2 coronavirus",
        "question_type": None,
    },
    {
        # Sample 5: Comparison — which came first
        "question": "Was the Eiffel Tower built before or after the Statue of Liberty?",
        "documents": [
            "Title: Eiffel Tower\nThe Eiffel Tower was constructed between 1887 and 1889 in Paris, France.",
            "Title: Statue of Liberty\nThe Statue of Liberty was a gift from France to the United States, dedicated on October 28, 1886.",
            "Title: French Architecture\nFrance has a long tradition of iconic architectural structures built in the 19th century.",
            "Title: American Landmarks\nThe United States has many famous landmarks gifted or built in the 19th century.",
        ],
        "gold_answer": "after",
        "question_type": "comparison",
    },
]


def run_demo():
    from pipeline import SelfHealingRAGPipeline
    from evaluation.metrics import exact_match, token_f1

    print("=" * 62)
    print("  Self-Healing Agentic RAG — Demo")
    print("=" * 62)

    pipeline   = SelfHealingRAGPipeline()
    total_em   = 0
    total_f1   = 0
    n_failures = 0
    n_corrected = 0

    for i, sample in enumerate(DEMO_SAMPLES, 1):
        print(f"\n{'─'*62}")
        print(f"[Sample {i}] {sample['question']}")
        print(f"Gold:      {sample['gold_answer']}")

        output = pipeline.run(
            question=sample["question"],
            documents=sample["documents"],
            question_type=sample.get("question_type"),
        )

        em = exact_match(output["final_answer"], sample["gold_answer"])
        f1 = token_f1(output["final_answer"], sample["gold_answer"])
        total_em += em
        total_f1 += f1
        if output["failure_detected"]:
            n_failures += 1
        if output["correction_attempted"] and em > 0:
            n_corrected += 1

        status = "✓" if em == 1 else ("~" if f1 > 0.3 else "✗")
        print(f"Predicted: {output['final_answer']}  {status}")
        print(f"EM={em:.0f}  F1={f1:.2f}  |  Failure={output['failure_detected']}  |  Corrections={output['n_correction_rounds']}")

        if output["failure_detected"]:
            print(f"  Failure reason : {output['failure_reason']}")

        print("Trace:")
        for step in output["trace"]:
            print(f"  -> {step}")

    n = len(DEMO_SAMPLES)
    print(f"\n{'='*62}")
    print(f"  FINAL SUMMARY ({n} samples)")
    print(f"  EM              : {total_em/n:.2f}  ({int(total_em)}/{n} correct)")
    print(f"  F1              : {total_f1/n:.2f}")
    print(f"  Failures found  : {n_failures}/{n}")
    print(f"  Corrections OK  : {n_corrected}")
    print(f"{'='*62}")


if __name__ == "__main__":
    run_demo()
