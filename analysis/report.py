"""
analysis/report.py — Markdown/LaTeX tables and plots from a results summary.

    python -m analysis.report results/hotpotqa_n200_s42          # print markdown, write plots
    python -m analysis.report results/hotpotqa_n200_s42 --latex  # also print a LaTeX table

Only numbers that exist in the summary are shown — missing values render as "–"
(the old plotting code silently substituted random numbers).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

DASH = "–"


def _f(v: Any, pct: bool = False, digits: int = 1) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return DASH
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{v * 100:.{digits}f}" if pct else f"{v:.2f}"
    return str(v)


def _ci(m: Dict[str, Any], key: str) -> str:
    if key not in m:
        return DASH
    lo, hi = m.get(f"{key}_ci", [None, None])
    base = _f(m[key], pct=True)
    return f"{base} ({_f(lo, True)}–{_f(hi, True)})" if lo is not None else base


def qa_table(summary: Dict[str, Any]) -> str:
    systems = summary.get("systems", {})
    comps = summary.get("comparisons", {})
    ref = summary.get("reference", "")
    long_answers = summary.get("long_answers", False)
    head = ["System", "EM % (95% CI)" if not long_answers else "F1 % (95% CI)",
            "F1 %" if not long_answers else "ROUGE-L %", "Support recall %", "LLM calls / q",
            "Latency s / q", f"Δ vs {ref}", "p-value"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    for name, m in systems.items():
        c = comps.get(name, {})
        delta_key, p_key = ("delta_em", "em_mcnemar_p") if not long_answers else ("delta_f1", "f1_bootstrap_p")
        delta = c.get(delta_key)
        delta_s = DASH if delta is None else f"{delta * 100:+.1f}"
        p_s = DASH if c.get(p_key) is None else f"{c[p_key]:.3f}"
        lines.append("| " + " | ".join([
            f"**{name}**" if name == "self_healing" else name,
            _ci(m, "em" if not long_answers else "f1"),
            _f(m.get("f1" if not long_answers else "rouge_l"), pct=True),
            _f(m.get("support_recall"), pct=True),
            _f(m.get("avg_llm_calls")),
            _f(m.get("avg_latency_s")),
            delta_s, p_s,
        ]) + " |")
    return "\n".join(lines)


def healing_table(summary: Dict[str, Any]) -> str:
    rows = [(n, m) for n, m in summary.get("systems", {}).items() if "healing" in m]
    if not rows:
        return ""
    head = ["System", "Monitor precision %", "Monitor recall %", "Wrong→right (fixed)",
            "Right→wrong (broken)", "Net gain (pp)", "Verified %", "EM verified %", "EM unverified %"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    for name, m in rows:
        h = m["healing"]
        lines.append("| " + " | ".join([
            name, _f(h["monitor_precision"], True), _f(h["monitor_recall"], True),
            str(h["fixed"]), str(h["broken"]), f"{h['net_gain'] * 100:+.1f}",
            _f(m.get("verified_rate"), True), _f(m.get("em_when_verified"), True),
            _f(m.get("em_when_unverified"), True),
        ]) + " |")
    return "\n".join(lines)


def tier_table(stats: Dict[str, Dict[str, Any]], key: str = "tier_distribution") -> str:
    rows = [(n, m.get(key)) for n, m in stats.items() if m.get(key)]
    if not rows:
        return ""
    lines = ["| System | No check | Tier 1 heuristics | Tier 2 NLI | Tier 3 LLM judge |",
             "|---|---:|---:|---:|---:|"]
    for name, d in rows:
        lines.append(f"| {name} | " + " | ".join(_f(d.get(f"tier_{t}", 0), True) for t in range(4)) + " |")
    return "\n".join(lines)


def detector_table(summary: Dict[str, Any]) -> str:
    head = ["Monitor", "Precision %", "Recall %", "F1 %", "Balanced acc. %", "AUROC", "LLM calls / response"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    for mode, m in summary.get("monitors", {}).items():
        lines.append("| " + " | ".join([
            f"**{mode}**" if mode == "tiered" else mode,
            _f(m.get("precision"), True), _f(m.get("recall"), True), _f(m.get("f1"), True),
            _f(m.get("balanced_accuracy"), True), _f(m.get("auroc")), _f(m.get("avg_llm_calls")),
        ]) + " |")
    return "\n".join(lines)


def render_markdown(summary: Dict[str, Any]) -> str:
    meta = summary.get("meta", {})
    meta_line = ", ".join(f"{k}={v}" for k, v in meta.items())
    parts = [f"## Results: `{summary.get('tag', '')}`", f"_{meta_line}_" if meta_line else ""]
    if summary.get("mode") == "detector":
        n = next(iter(summary.get("monitors", {}).values()), {}).get("n_samples", 0)
        parts += [f"Hallucination detection on {n} labelled responses "
                  "(positive class = response NOT fully grounded).", "", detector_table(summary), "",
                  "### Which tier made the decision", "", tier_table(summary.get("monitors", {}))]
    else:
        n = next(iter(summary.get("systems", {}).values()), {}).get("n_samples", 0)
        parts += [f"Answer quality on {n} questions.", "", qa_table(summary)]
        heal = healing_table(summary)
        if heal:
            parts += ["", "### Self-healing analysis", "",
                      "Precision/recall: how well the monitor's flag predicts that the *initial* answer is wrong. "
                      "Verified: the returned answer passed the monitor; a useful monitor gives much higher EM on "
                      "verified than on unverified answers.",
                      "", heal]
        tiers = tier_table(summary.get("systems", {}))
        if tiers:
            parts += ["", "### Deciding monitor tier (highest tier reached per question)", "", tiers]
    return "\n".join(p for p in parts if p is not None) + "\n"


def to_latex(summary: Dict[str, Any]) -> str:
    rows = []
    for name, m in summary.get("systems", {}).items():
        rows.append(f"{name.replace('_', ' ')} & {_f(m.get('em'), True)} & {_f(m.get('f1'), True)} & "
                    f"{_f(m.get('avg_llm_calls'))} \\\\")
    return "\n".join([
        r"\begin{tabular}{lccc}", r"\toprule", r"System & EM & F1 & LLM calls/q \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}",
    ])


def make_plots(summary: Dict[str, Any], out_dir: str) -> List[str]:
    """Bar chart of accuracy vs LLM cost per system (QA) or detector F1 (detector)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    if summary.get("mode") == "detector":
        mon = summary.get("monitors", {})
        names = list(mon)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(names, [mon[n].get("f1", 0) * 100 for n in names], color="#4C72B0")
        ax.set_ylabel("Hallucination-detection F1 (%)")
        for i, n in enumerate(names):
            ax.annotate(f"{mon[n].get('avg_llm_calls', 0):.2f} calls", (i, mon[n].get("f1", 0) * 100),
                        ha="center", va="bottom", fontsize=8)
        ax.set_title("Detector quality vs LLM calls per response")
    else:
        sysm = summary.get("systems", {})
        names = [n for n in sysm if "em" in sysm[n] or "f1" in sysm[n]]
        key = "f1" if summary.get("long_answers") else "em"
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = [sysm[n].get("avg_llm_calls", 0) for n in names]
        ys = [sysm[n].get(key, 0) * 100 for n in names]
        ax.scatter(xs, ys, s=60, color="#4C72B0")
        for x, y, n in zip(xs, ys, names, strict=False):
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("Average LLM calls per question")
        ax.set_ylabel(f"{key.upper()} (%)")
        ax.set_title("Accuracy vs cost")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "summary.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)
    return paths


def load_summary(path: str) -> Dict[str, Any]:
    if os.path.isdir(path):
        path = os.path.join(path, "summary.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Render tables/plots from a results directory.")
    ap.add_argument("path", help="results/<tag> directory or a summary.json file")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    summary = load_summary(args.path)
    print(render_markdown(summary))
    if args.latex and summary.get("mode") != "detector":
        print(to_latex(summary))
    if not args.no_plots:
        out = args.path if os.path.isdir(args.path) else os.path.dirname(args.path)
        for p in make_plots(summary, out):
            print(f"plot: {p}")


if __name__ == "__main__":
    main()
