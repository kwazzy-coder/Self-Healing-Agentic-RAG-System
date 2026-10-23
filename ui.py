"""
ui.py — Gradio web interface.

    python ui.py        # then open http://localhost:7860

Shows the answer, whether the monitor flagged it, every self-healing round
(answer, deciding tier, confidence) and the passages that were used.
"""
from __future__ import annotations

import threading
import time
from typing import Any, List, Optional

import gradio as gr

from evaluation.metrics import exact_match, token_f1

_PIPELINE = None
_LOCK = threading.Lock()


def get_pipeline() -> Any:
    """Load models on first use (so the page opens instantly)."""
    global _PIPELINE
    with _LOCK:
        if _PIPELINE is None:
            from pipeline import SelfHealingRAGPipeline

            _PIPELINE = SelfHealingRAGPipeline()
    return _PIPELINE


DEMO_DOCS = {
    "Eiffel Tower": "Title: Eiffel Tower\nThe Eiffel Tower is a wrought-iron lattice tower located in Paris, France. "
                    "Built in 1889 by Gustave Eiffel for the 1889 World's Fair.",
    "France": "Title: France\nFrance is a country in Western Europe. Its capital and largest city is Paris.",
    "Paris": "Title: Paris\nParis is the capital and most populous city of France, situated on the River Seine.",
    "COVID-19": "Title: COVID-19\nCOVID-19 is caused by the SARS-CoV-2 coronavirus, first identified in Wuhan, "
                "China in December 2019.",
    "Inception": "Title: Inception\nInception is a 2010 science fiction film written and directed by Christopher "
                 "Nolan. The film stars Leonardo DiCaprio.",
    "Christopher Nolan": "Title: Christopher Nolan\nChristopher Nolan is a British-American filmmaker born in "
                         "London, England on July 30, 1970.",
    "Statue of Liberty": "Title: Statue of Liberty\nThe Statue of Liberty was a gift from France to the United "
                         "States, dedicated on October 28, 1886.",
    "Germany": "Title: Germany\nGermany is a country in Central Europe. Its capital is Berlin.",
}

PRESETS = {
    "Multi-hop (bridge)": ("What is the capital of the country where the Eiffel Tower is located?",
                           ["Eiffel Tower", "France", "Paris", "Germany"], "Paris", "bridge"),
    "Movie director nationality": ("What is the nationality of the director of Inception?",
                                   ["Inception", "Christopher Nolan", "Germany"], "British", "bridge"),
    "Comparison": ("Was the Eiffel Tower built before or after the Statue of Liberty?",
                   ["Eiffel Tower", "Statue of Liberty", "France"], "after", "comparison"),
    "Answer not in documents": ("Who invented the telephone?", ["Germany", "France", "COVID-19"],
                                "Alexander Graham Bell", "auto-detect"),
    "Simple factual": ("What causes COVID-19?", ["COVID-19", "France"], "SARS-CoV-2 coronavirus", "auto-detect"),
}


def load_preset(name: str):
    q, keys, gold, qt = PRESETS[name]
    return q, "\n\n".join(DEMO_DOCS[k] for k in keys), gold, qt


def run_query(question: str, doc_text: str, gold: str, qtype: str):
    if not question.strip():
        return "Please enter a question.", "", [], "", ""
    docs: List[str] = [d.strip() for d in doc_text.split("\n\n") if d.strip()] or list(DEMO_DOCS.values())
    qt: Optional[str] = None if qtype == "auto-detect" else qtype
    start = time.perf_counter()
    try:
        out = get_pipeline().run(question, docs, question_type=qt)
    except Exception as err:  # show provider errors in the UI instead of a stack trace
        return f"Error: {err}", "", [], "", ""
    elapsed = time.perf_counter() - start

    healed = out.selected_round > 0
    status = [
        f"Strategy: {out.strategy}",
        f"Monitor: {'flagged ' + out.failure_type if out.failure_detected else 'passed'}",
        f"Self-healing: {'answer replaced (round ' + str(out.selected_round) + ')' if healed else 'not needed'}",
        f"Cost: {out.cost_data.get('n_llm_calls', 0)} LLM calls, {elapsed:.1f}s",
    ]
    if gold.strip():
        em, f1 = exact_match(out.final_answer, gold), token_f1(out.final_answer, gold)
        status.append(f"vs gold: EM={em:.0f}  F1={f1:.2f}")
    rounds = [[r.round, r.produced_by, r.answer,
               "" if r.has_failure is None else ("FAIL " + r.failure_type if r.has_failure else "PASS"),
               r.monitor_tier, "" if r.confidence is None else f"{r.confidence:.2f}"] for r in out.rounds]
    passages = "\n\n".join(f"[{i}] {d[:400]}" for i, d in enumerate(out.retrieved_docs, 1))
    return out.final_answer, "\n".join(status), rounds, "\n".join(out.trace), passages


with gr.Blocks(title="Self-Healing Agentic RAG") as demo:
    gr.Markdown(
        "# Self-Healing Agentic RAG\n"
        "A RAG system that checks its own answers (heuristics → NLI → LLM judge) and repairs them "
        "(targeted re-retrieval, decomposition, reformulation) before replying."
    )
    with gr.Row():
        preset = gr.Dropdown(list(PRESETS), value="Movie director nationality", label="Load an example")
        load_btn = gr.Button("Load")
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Question", lines=2)
            documents = gr.Textbox(label="Documents (separate with a blank line)", lines=10)
            with gr.Row():
                gold = gr.Textbox(label="Gold answer (optional)")
                qtype = gr.Dropdown(["auto-detect", "direct", "bridge", "comparison"], value="auto-detect",
                                    label="Question type")
            run_btn = gr.Button("Run", variant="primary")
        with gr.Column():
            answer = gr.Textbox(label="Final answer", lines=2)
            status = gr.Textbox(label="Self-healing status", lines=6)
            rounds = gr.Dataframe(headers=["round", "produced by", "answer", "monitor", "tier", "confidence"],
                                  label="Self-healing rounds", wrap=True)
    with gr.Row():
        trace = gr.Textbox(label="Trace", lines=10)
        passages = gr.Textbox(label="Passages used", lines=10)

    load_btn.click(load_preset, inputs=[preset], outputs=[question, documents, gold, qtype])
    run_btn.click(run_query, inputs=[question, documents, gold, qtype],
                  outputs=[answer, status, rounds, trace, passages])
    demo.load(lambda: load_preset("Movie director nationality"), outputs=[question, documents, gold, qtype])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
