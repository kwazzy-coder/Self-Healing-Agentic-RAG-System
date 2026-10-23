"""
evaluation/benchmark_retrieval.py — Compare BM25, dense, linear fusion and RRF
on HotpotQA (distractor) — no LLM or API key needed.

    python -m evaluation.benchmark_retrieval --n_samples 500

Each question has 10 candidate paragraphs, 2 of which are the gold
"supporting" paragraphs. For multi-hop QA what matters is whether *both* gold
paragraphs are retrieved, so `all_gold@k` is reported next to recall and MRR.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Sequence

import config
from evaluation.metrics import all_gold_at_k, retrieval_mrr, retrieval_recall_at_k
from evaluation.statistical import bootstrap_confidence_interval
from retrieval.retriever import HybridRetriever, extract_title

METHODS = ("bm25", "dense", "linear", "rrf")
KS = (2, 5)


def benchmark(samples: Sequence[Dict[str, Any]], retriever: HybridRetriever,
              methods: Sequence[str] = METHODS) -> Dict[str, Dict[str, Any]]:
    per_method: Dict[str, Dict[str, List[float]]] = {m: {} for m in methods}
    timings = {m: 0.0 for m in methods}
    for s in samples:
        docs = [f"Title: {d['title']}\n{d['text']}" for d in s["context_docs"]]
        index = retriever.build_index(docs)
        gold = s["supporting_titles"]
        for m in methods:
            t0 = time.perf_counter()
            titles = [extract_title(r.text) for r in index.search(s["question"], top_k=len(docs), method=m)]
            timings[m] += time.perf_counter() - t0
            row = per_method[m]
            for k in KS:
                row.setdefault(f"recall@{k}", []).append(retrieval_recall_at_k(titles, gold, k))
                row.setdefault(f"all_gold@{k}", []).append(all_gold_at_k(titles, gold, k))
            row.setdefault("mrr", []).append(retrieval_mrr(titles, gold))
    out = {}
    for m, metrics in per_method.items():
        out[m] = {}
        for name, vals in metrics.items():
            mean, lo, hi = bootstrap_confidence_interval(vals)
            out[m][name] = mean
            out[m][f"{name}_ci"] = [lo, hi]
        out[m]["ms_per_query"] = 1000 * timings[m] / max(1, len(samples))
    return out


def to_markdown(results: Dict[str, Dict[str, Any]], n: int) -> str:
    cols = [f"recall@{k}" for k in KS] + [f"all_gold@{k}" for k in KS] + ["mrr"]
    lines = [f"Retrieval on {n} HotpotQA (distractor) questions — 10 candidates, 2 gold.", "",
             "| Method | " + " | ".join(cols) + " |", "|---|" + "|".join("---:" for _ in cols) + "|"]
    labels = {"bm25": "BM25", "dense": "Dense (MiniLM)", "linear": "Hybrid (0.3·BM25 + 0.7·dense)",
              "rrf": "Hybrid (RRF)"}
    for m, r in results.items():
        lines.append(f"| {labels.get(m, m)} | " + " | ".join(f"{100 * r[c]:.1f}" for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = ap.parse_args()

    from data.hotpotqa_loader import load_hotpotqa

    samples = load_hotpotqa(args.n_samples, seed=args.seed)
    results = benchmark(samples, HybridRetriever())
    md = to_markdown(results, len(samples))
    out_dir = os.path.join(config.RESULTS_DIR, f"retrieval_n{len(samples)}_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_samples": len(samples), "seed": args.seed, "embedding_model": config.EMBEDDING_MODEL,
                   "results": results}, fh, indent=2)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    print(md)
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
