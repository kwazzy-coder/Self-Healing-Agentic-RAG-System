"""
evaluation/metrics.py — QA, retrieval and detection metrics.

Answer EM/F1 follow the *official* HotpotQA evaluation script (lower-case,
strip punctuation and articles, yes/no/noanswer special case for F1), so the
numbers are comparable with published results.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """Official SQuAD/HotpotQA normalisation."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 (HotpotQA variant: yes/no/noanswer must match exactly)."""
    pred_n, gold_n = normalize_answer(prediction), normalize_answer(gold)
    special = {"yes", "no", "noanswer"}
    if (pred_n in special or gold_n in special) and pred_n != gold_n:
        return 0.0
    pred_t, gold_t = pred_n.split(), gold_n.split()
    if not pred_t or not gold_t:
        return float(pred_t == gold_t)
    common = Counter(pred_t) & Counter(gold_t)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred_t), same / len(gold_t)
    return 2 * precision * recall / (precision + recall)


def contains_match(prediction: str, gold: str) -> float:
    """Lenient match: gold appears inside the prediction (or vice versa) with similar length."""
    p, g = normalize_answer(prediction), normalize_answer(gold)
    if p == g:
        return 1.0
    if p and g and (g in p or p in g) and abs(len(p.split()) - len(g.split())) <= 2:
        return 1.0
    return 0.0


def best_over_golds(metric, prediction: str, golds: Iterable[str]) -> float:
    """Max of a metric over a gold answer and its aliases (used for MuSiQue)."""
    golds = [g for g in golds if g]
    return max((metric(prediction, g) for g in golds), default=0.0)


def rouge_l(prediction: str, gold: str) -> float:
    """ROUGE-L F-measure (0.0 if `rouge-score` is not installed)."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(gold, prediction)["rougeL"].fmeasure


def bert_score_batch(predictions: Sequence[str], golds: Sequence[str]) -> List[float]:
    """BERTScore F1 for many pairs at once (slow; downloads roberta-large on first use)."""
    from bert_score import score as bert_score

    if not predictions:
        return []
    preds = [p if p.strip() else "-" for p in predictions]
    _, _, f1 = bert_score(list(preds), list(golds), lang="en", verbose=False)
    return [float(x) for x in f1]


# ── Retrieval ────────────────────────────────────────────────────────────────
def _norm_title(t: str) -> str:
    return " ".join(t.lower().split())


def supporting_doc_f1(retrieved_titles: Sequence[str], gold_titles: Sequence[str]) -> float:
    """F1 between the set of passages given to the generator and the gold supporting passages."""
    gold = {_norm_title(t) for t in gold_titles if t}
    got = {_norm_title(t) for t in retrieved_titles if t}
    if not gold:
        return 1.0
    hit = len(gold & got)
    if hit == 0:
        return 0.0
    precision, recall = hit / len(got), hit / len(gold)
    return 2 * precision * recall / (precision + recall)


def retrieval_recall_at_k(retrieved_titles: Sequence[str], gold_titles: Sequence[str],
                          k: Optional[int] = None) -> float:
    gold = {_norm_title(t) for t in gold_titles if t}
    if not gold:
        return 1.0
    got = {_norm_title(t) for t in (retrieved_titles[:k] if k else retrieved_titles)}
    return len(gold & got) / len(gold)


def all_gold_at_k(retrieved_titles: Sequence[str], gold_titles: Sequence[str], k: int) -> float:
    """1.0 only if *every* gold passage is in the top-k (what multi-hop QA actually needs)."""
    return float(retrieval_recall_at_k(retrieved_titles, gold_titles, k) == 1.0)


def retrieval_mrr(retrieved_titles: Sequence[str], gold_titles: Sequence[str]) -> float:
    gold = {_norm_title(t) for t in gold_titles if t}
    if not gold:
        return 1.0
    for rank, t in enumerate(retrieved_titles, start=1):
        if _norm_title(t) in gold:
            return 1.0 / rank
    return 0.0


# ── Binary detection ─────────────────────────────────────────────────────────
def binary_classification_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if not t and not p)
    n = max(1, len(y_true))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def roc_auc(y_true: Sequence[bool], scores: Sequence[float]) -> Optional[float]:
    """AUROC via the Mann-Whitney U statistic (ties count half). None if one class is missing."""
    pos = [s for t, s in zip(y_true, scores, strict=False) if t]
    neg = [s for t, s in zip(y_true, scores, strict=False) if not t]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(pos) * len(neg))


# ── Pretty printing ──────────────────────────────────────────────────────────
def print_metrics(metrics: Dict, title: str = "Results") -> None:
    print(f"\n{'=' * 56}\n  {title}\n{'=' * 56}")
    for k, v in metrics.items():
        if isinstance(v, dict):
            continue
        print(f"  {k:<36}: {v:.4f}" if isinstance(v, float) else f"  {k:<36}: {v}")
    print("=" * 56)


def format_results_table(rows: List[tuple], columns: Sequence[str] = ("em", "f1", "avg_llm_calls")) -> str:
    """Markdown table from [(system_name, metrics_dict), ...]."""
    header = "| System | " + " | ".join(columns) + " |"
    sep = "|---|" + "|".join("---:" for _ in columns) + "|"
    lines = [header, sep]
    for name, m in rows:
        cells = []
        for c in columns:
            v = m.get(c)
            cells.append(f"{v:.3f}" if isinstance(v, float) else ("-" if v is None else str(v)))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
