import json
import os

from agent.self_monitor import MonitorSettings, SelfMonitor
from analysis.error_analysis import analyze, case_studies
from analysis.report import make_plots, render_markdown
from baselines import build_system
from evaluation.evaluator import Evaluator
from pipeline import Components
from tests.conftest import DOCS
from tests.fakes import FakeChatClient, make_llm

SAMPLES = [
    {"id": "q1", "question": "What is the nationality of the director of Inception?", "answer": "British",
     "context_docs": [{"title": d.split("\n")[0][7:], "text": d.split("\n", 1)[1]} for d in DOCS],
     "supporting_titles": ["Inception", "Christopher Nolan"], "type": "bridge"},
    {"id": "q2", "question": "What is the capital of France?", "answer": "Paris",
     "context_docs": [{"title": d.split("\n")[0][7:], "text": d.split("\n", 1)[1]} for d in DOCS],
     "supporting_titles": ["France"], "type": "comparison"},
]


def _components(retriever, nli, client):
    return Components(llm=make_llm(client), retriever=retriever, nli=nli)


def test_qa_evaluation_end_to_end(tmp_path, retriever, nli):
    client = FakeChatClient(answers={"nationality of the director": ["German", "British"],
                                     "capital of France": "Paris"})
    comp = _components(retriever, nli, client)
    systems = {"vanilla_rag": build_system("vanilla_rag", comp),
               "self_healing": build_system("self_healing", comp, enable_planning=False)}
    # vanilla consumes "German" first; self_healing then sees "British" -> both deterministic per order
    summary = Evaluator(str(tmp_path)).evaluate_qa(systems, SAMPLES, tag="t")
    sh = summary["systems"]["self_healing"]
    assert summary["reference"] == "vanilla_rag"
    assert sh["n_samples"] == 2 and 0 <= sh["em"] <= 1 and "em_ci" in sh
    assert "self_healing" in summary["comparisons"]
    assert "healing" in sh and "monitor_precision" in sh["healing"]
    assert os.path.exists(tmp_path / "t" / "report.md")
    md = render_markdown(summary)
    assert "self_healing" in md and "Self-healing analysis" in md
    recs = [json.loads(line) for line in open(tmp_path / "t" / "self_healing.jsonl")]
    assert analyze(recs)["n"] == 2
    assert isinstance(case_studies(recs, "fixed"), str)
    assert make_plots(summary, str(tmp_path / "plots"))


def test_resume_skips_finished_and_retries_errors(tmp_path, retriever, nli):
    client = FakeChatClient(answers={"capital of France": "Paris", "nationality": "British"})
    calls = {"n": 0}

    class Flaky:
        def __init__(self, inner):
            self.inner = inner

        def run(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("provider down")
            return self.inner.run(*a, **k)

    sys_ = Flaky(build_system("vanilla_rag", _components(retriever, nli, client)))
    ev = Evaluator(str(tmp_path))
    first = ev.evaluate_qa({"vanilla_rag": sys_}, SAMPLES, tag="r")
    assert first["systems"]["vanilla_rag"]["n_errors"] == 1
    second = ev.evaluate_qa({"vanilla_rag": sys_}, SAMPLES, tag="r")
    assert second["systems"]["vanilla_rag"]["n_errors"] == 0 and calls["n"] == 3  # only the failed one re-ran


def test_detector_evaluation(tmp_path, nli):
    samples = [
        {"id": "a", "question": "What is the capital of France?", "gold_response": "Paris.",
         "documents": DOCS, "adherence_score": True},
        {"id": "b", "question": "What is the capital of France?", "gold_response": "Lyon.",
         "documents": DOCS, "adherence_score": False},
    ]
    monitors = {m: SelfMonitor(make_llm(FakeChatClient()), nli=nli, settings=MonitorSettings.from_mode(m))
                for m in ("heuristic", "heuristic_nli", "tiered")}
    summary = Evaluator(str(tmp_path)).evaluate_detector(monitors, samples, tag="d")
    assert summary["monitors"]["heuristic_nli"]["f1"] == 1.0
    assert summary["monitors"]["heuristic_nli"]["auroc"] == 1.0
    assert summary["monitors"]["heuristic"]["recall"] == 0.0     # heuristics alone can't see this
    assert "AUROC" in render_markdown(summary)
