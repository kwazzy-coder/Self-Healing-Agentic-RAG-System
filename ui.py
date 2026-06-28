"""
ui.py — Gradio Web Interface for Self-Healing RAG
Run: python ui.py
Then open: http://localhost:7860
"""

import gradio as gr
import json
import time

# ── Load pipeline once ────────────────────────────────────────────────────────
print("Loading pipeline...")
from pipeline import SelfHealingRAGPipeline
from evaluation.metrics import exact_match, token_f1

pipeline = SelfHealingRAGPipeline()
print("Pipeline ready!")

# ── Demo documents for quick testing ──────────────────────────────────────────
DEMO_DOCS = {
    "Eiffel Tower": "Title: Eiffel Tower\nThe Eiffel Tower is a wrought-iron lattice tower located in Paris, France. Built in 1889 by Gustave Eiffel for the 1889 World's Fair.",
    "France": "Title: France\nFrance is a country in Western Europe. Its capital and largest city is Paris. The official language is French.",
    "Paris": "Title: Paris\nParis is the capital and most populous city of France, situated on the River Seine. It is known as the City of Light.",
    "COVID-19": "Title: COVID-19\nCOVID-19 is caused by the SARS-CoV-2 coronavirus, first identified in Wuhan, China in December 2019.",
    "Inception": "Title: Inception\nInception is a 2010 science fiction film written and directed by Christopher Nolan. The film stars Leonardo DiCaprio.",
    "Christopher Nolan": "Title: Christopher Nolan\nChristopher Nolan is a British-American filmmaker born in London, England on July 30, 1970.",
    "Statue of Liberty": "Title: Statue of Liberty\nThe Statue of Liberty was a gift from France to the United States, dedicated on October 28, 1886.",
    "Germany": "Title: Germany\nGermany is a country in Central Europe. Its capital is Berlin. The official language is German.",
}


def run_query(question, doc_text, gold_answer, question_type):
    """Run the self-healing RAG pipeline and return formatted results."""
    if not question.strip():
        return "Please enter a question.", "", "", ""

    # Parse documents — split by newlines if multiple provided
    if doc_text.strip():
        docs = [d.strip() for d in doc_text.strip().split("\n\n") if d.strip()]
    else:
        # Use all demo docs if none provided
        docs = list(DEMO_DOCS.values())

    qt = question_type if question_type != "auto-detect" else None

    start = time.time()
    output = pipeline.run(
        question=question,
        documents=docs,
        question_type=qt,
    )
    elapsed = time.time() - start

    # Format answer
    answer = output["final_answer"]

    # Format metrics
    metrics_lines = [
        f"⏱️ Time taken: {elapsed:.1f}s",
        f"📄 Documents retrieved: {len(output['retrieved_docs'])}",
        f"🔍 Failure detected: {'Yes ⚠️' if output['failure_detected'] else 'No ✅'}",
        f"🔧 Corrections applied: {output['n_correction_rounds']}",
    ]
    if output["failure_detected"]:
        metrics_lines.append(f"❌ Failure reason: {output['failure_reason']}")
    if gold_answer.strip():
        em = exact_match(answer, gold_answer)
        f1 = token_f1(answer, gold_answer)
        metrics_lines.append(f"✓ Exact Match: {'Yes ✅' if em else 'No ❌'}")
        metrics_lines.append(f"📊 F1 Score: {f1:.2f}")

    metrics_str = "\n".join(metrics_lines)

    # Format trace
    trace_str = "\n".join([f"  → {step}" for step in output["trace"]])

    # Format retrieved docs
    docs_str = "\n\n".join([
        f"[Doc {i+1}]\n{d[:300]}..."
        for i, d in enumerate(output["retrieved_docs"][:3])
    ])

    return answer, metrics_str, trace_str, docs_str


def load_demo(demo_name):
    """Load a preset demo question."""
    demos = {
        "Capital (Simple)": (
            "What is the capital of France?",
            "\n\n".join([DEMO_DOCS["France"], DEMO_DOCS["Paris"], DEMO_DOCS["Germany"]]),
            "Paris",
            "direct"
        ),
        "Multi-hop (Bridge)": (
            "What is the capital of the country where the Eiffel Tower is located?",
            "\n\n".join([DEMO_DOCS["Eiffel Tower"], DEMO_DOCS["France"], DEMO_DOCS["Paris"]]),
            "Paris",
            "bridge"
        ),
        "Movie Director": (
            "What is the nationality of the director of Inception?",
            "\n\n".join([DEMO_DOCS["Inception"], DEMO_DOCS["Christopher Nolan"]]),
            "British",
            "bridge"
        ),
        "COVID Cause": (
            "What causes COVID-19?",
            DEMO_DOCS["COVID-19"],
            "SARS-CoV-2 coronavirus",
            "direct"
        ),
        "Comparison": (
            "Was the Eiffel Tower built before or after the Statue of Liberty?",
            "\n\n".join([DEMO_DOCS["Eiffel Tower"], DEMO_DOCS["Statue of Liberty"]]),
            "after",
            "comparison"
        ),
    }
    if demo_name in demos:
        q, docs, gold, qt = demos[demo_name]
        return q, docs, gold, qt
    return "", "", "", "direct"


# ── Build Gradio UI ───────────────────────────────────────────────────────────
with gr.Blocks(
    title="Self-Healing RAG System",
    theme=gr.themes.Soft(),
    css="""
    .header { text-align: center; padding: 20px; }
    .answer-box { font-size: 1.4em; font-weight: bold; color: #2196F3; }
    """
) as demo:

    gr.HTML("""
    <div class="header">
        <h1>🔄 Self-Healing Agentic RAG System</h1>
        <p>A RAG system that detects its own failures and automatically corrects them</p>
        <p><b>Team:</b> Samrudha Narendra · K Shaman · ISL Nandini | PES University</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🚀 Quick Demo")
            demo_dropdown = gr.Dropdown(
                choices=["Capital (Simple)", "Multi-hop (Bridge)",
                         "Movie Director", "COVID Cause", "Comparison"],
                label="Load a preset example",
                value="Multi-hop (Bridge)"
            )
            load_btn = gr.Button("Load Example", variant="secondary")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Input")
            question_input = gr.Textbox(
                label="Question",
                placeholder="e.g. What is the nationality of the director of Inception?",
                lines=2
            )
            doc_input = gr.Textbox(
                label="Documents (separate multiple docs with blank line)",
                placeholder="Title: Inception\nInception is a 2010 film directed by Christopher Nolan...\n\nTitle: Christopher Nolan\nChristopher Nolan is a British filmmaker...",
                lines=8
            )
            with gr.Row():
                gold_input = gr.Textbox(
                    label="Gold Answer (optional — for scoring)",
                    placeholder="e.g. British",
                    scale=2
                )
                qt_input = gr.Dropdown(
                    choices=["auto-detect", "direct", "bridge", "comparison"],
                    label="Question Type",
                    value="auto-detect",
                    scale=1
                )
            run_btn = gr.Button("🔍 Run Self-Healing RAG", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Output")
            answer_output = gr.Textbox(
                label="Final Answer",
                lines=2,
                elem_classes=["answer-box"]
            )
            metrics_output = gr.Textbox(
                label="Metrics & Self-Healing Status",
                lines=8,
                interactive=False
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔍 Reasoning Trace (step-by-step)")
            trace_output = gr.Textbox(
                label="",
                lines=10,
                interactive=False
            )
        with gr.Column():
            gr.Markdown("### 📄 Retrieved Documents")
            docs_output = gr.Textbox(
                label="",
                lines=10,
                interactive=False
            )

    gr.Markdown("""
    ---
    ### How Self-Healing Works
    1. **Task Planner** — detects if question needs multi-hop reasoning
    2. **Hybrid Retriever** — BM25 + FAISS semantic search (top-7 docs)
    3. **LLM Reasoner** — Gemini generates concise answer
    4. **Self-Monitor** — 4 checks: confidence, grounding, completeness, LLM verification
    5. **Corrective Actions** — if failure: re-retrieve → reformulate → decompose (max 3 rounds)
    """)

    # ── Event handlers ────────────────────────────────────────────────────────
    run_btn.click(
        fn=run_query,
        inputs=[question_input, doc_input, gold_input, qt_input],
        outputs=[answer_output, metrics_output, trace_output, docs_output]
    )

    load_btn.click(
        fn=load_demo,
        inputs=[demo_dropdown],
        outputs=[question_input, doc_input, gold_input, qt_input]
    )

    # Load default demo on start
    demo.load(
        fn=lambda: load_demo("Multi-hop (Bridge)"),
        outputs=[question_input, doc_input, gold_input, qt_input]
    )


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Self-Healing RAG — Web Interface")
    print("  Open: http://localhost:7860")
    print("="*50 + "\n")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
