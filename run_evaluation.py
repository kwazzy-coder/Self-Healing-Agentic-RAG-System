"""
run_evaluation.py — Command-line entry point for all evaluations.

Examples
--------
# Self-healing vs vanilla RAG on 100 HotpotQA questions (paired significance tests included)
python run_evaluation.py --dataset hotpotqa --n_samples 100 --compare_baseline

# Any set of systems (see baselines/__init__.py)
python run_evaluation.py --dataset hotpotqa --n_samples 100 \
    --systems vanilla_rag,planner_only,self_healing,llm_judge_only

# Only bridge questions
python run_evaluation.py --dataset hotpotqa --n_samples 100 --question_type bridge --compare_baseline

# Hallucination-detector benchmark on RAGBench (heuristic+NLI needs NO API key)
python run_evaluation.py --dataset ragbench --subset covidqa --mode detector \
    --monitors heuristic,heuristic_nli,tiered

# Long-form QA on RAGBench
python run_evaluation.py --dataset ragbench --subset msmarco --mode qa --compare_baseline

Results go to results/<tag>/ (per-sample JSONL, summary.json, report.md).
Re-running the same command resumes where it stopped.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, List

import config

logger = logging.getLogger("run_evaluation")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-Healing Agentic RAG — evaluation")
    p.add_argument("--dataset", choices=["hotpotqa", "musique", "ragbench"], default="hotpotqa")
    p.add_argument("--subset", default="covidqa", help="RAGBench subset (covidqa, hotpotqa, msmarco, pubmedqa, ...)")
    p.add_argument("--setting", choices=["distractor", "fullwiki"], default="distractor")
    p.add_argument("--question_type", choices=["bridge", "comparison"], default=None)
    p.add_argument("--mode", choices=["qa", "detector"], default="qa")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--systems", default="self_healing",
                   help="comma-separated: vanilla_rag, planner_only, self_healing, nli_only, "
                        "llm_judge_only, always_correct, crag")
    p.add_argument("--compare_baseline", action="store_true", help="shortcut: add vanilla_rag as reference")
    p.add_argument("--monitors", default="heuristic,heuristic_nli,tiered",
                   help="detector mode: comma-separated monitor modes "
                        "(heuristic, nli, heuristic_nli, heuristic_llm, llm, tiered)")
    p.add_argument("--provider", default=None, help="override LLM_BACKEND (gemini, groq, openai, ollama, huggingface)")
    p.add_argument("--model", default=None, help="override the model id")
    p.add_argument("--tag", default=None, help="results sub-directory name")
    p.add_argument("--no_resume", action="store_true", help="start from scratch instead of resuming")
    return p.parse_args(argv)


def load_samples(args: argparse.Namespace) -> List[Dict]:
    if args.dataset == "hotpotqa":
        from data.hotpotqa_loader import load_hotpotqa

        return load_hotpotqa(args.n_samples, setting=args.setting, seed=args.seed, question_type=args.question_type)
    if args.dataset == "musique":
        from data.musique_loader import load_musique

        return load_musique(args.n_samples, seed=args.seed)
    from data.ragbench_loader import load_ragbench

    return load_ragbench(args.subset, split="test", n_samples=args.n_samples, seed=args.seed)


def default_tag(args: argparse.Namespace) -> str:
    name = args.dataset + (f"-{args.subset}" if args.dataset == "ragbench" else "")
    if args.question_type:
        name += f"-{args.question_type}"
    return f"{name}_{args.mode}_n{args.n_samples}_s{args.seed}"


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)
    if args.model:
        config.LLM_MODEL = args.model

    from agent.llm_reasoner import LLMReasoner
    from evaluation.evaluator import Evaluator
    from pipeline import Components

    samples = load_samples(args)
    if not samples:
        logger.error("No samples loaded.")
        return 1
    tag = args.tag or default_tag(args)
    evaluator = Evaluator()

    if args.mode == "detector":
        if args.dataset != "ragbench":
            logger.error("--mode detector needs labelled responses: use --dataset ragbench.")
            return 1
        from agent.nli import NLIScorer
        from agent.self_monitor import MonitorSettings, SelfMonitor

        modes = [m.strip() for m in args.monitors.split(",") if m.strip()]
        needs_llm = any(MonitorSettings.from_mode(m).use_llm_judge for m in modes)
        llm = LLMReasoner(args.provider) if needs_llm else None
        nli = NLIScorer()
        monitors = {m: SelfMonitor(llm, nli=nli, settings=MonitorSettings.from_mode(m, short_answers=False))
                    for m in modes}
        meta = {"dataset": f"ragbench/{args.subset}", "n": len(samples), "seed": args.seed,
                "llm": f"{llm.backend}/{llm.model}" if llm else "none", "nli": config.NLI_MODEL,
                "nli_band": f"[{config.NLI_LOW}, {config.NLI_HIGH}]"}
        summary = evaluator.evaluate_detector(monitors, samples, tag, resume=not args.no_resume, meta=meta)
    else:
        from baselines import build_system

        names = [s.strip() for s in args.systems.split(",") if s.strip()]
        if args.compare_baseline and "vanilla_rag" not in names:
            names.insert(0, "vanilla_rag")
        components = Components.build(llm=LLMReasoner(args.provider))
        long_answers = args.dataset == "ragbench"
        systems = {n: build_system(n, components, short_answers=not long_answers) for n in names}
        meta = {"dataset": args.dataset + (f"/{args.subset}" if args.dataset == "ragbench" else ""),
                "n": len(samples), "seed": args.seed, "llm": f"{components.llm.backend}/{components.llm.model}",
                "top_k": config.TOP_K_RETRIEVAL, "max_rounds": config.MAX_CORRECTION_ATTEMPTS}
        summary = evaluator.evaluate_qa(systems, samples, tag, long_answers=long_answers,
                                        resume=not args.no_resume, meta=meta)

    from analysis.report import render_markdown

    print("\n" + render_markdown(summary))
    print(f"Saved to {config.RESULTS_DIR}/{tag}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
