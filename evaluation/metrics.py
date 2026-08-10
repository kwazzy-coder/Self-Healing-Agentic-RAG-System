"""
evaluation/metrics.py

Implements rigorous evaluation metrics for Open-Domain QA and RAG systems.
Supports Exact Match, Token F1, ROUGE-L, BERTScore, and Retrieval metrics.
"""
import re
from collections import Counter
from typing import Dict, List, Optional

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    from bert_score import score as bert_score
    HAS_BERT_SCORE = True
except ImportError:
    HAS_BERT_SCORE = False


def normalize_answer(s: str) -> str:
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    if not isinstance(s, str):
        return ""
    
    s = s.lower().strip()

    # Remove punctuation except hyphens in compound words
    s = re.sub(r"[^\w\s-]", " ", s)

    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)

    # Normalize number words to digits
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16",
        "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
        "thirty": "30", "forty": "40", "fifty": "50", "hundred": "100",
        "thousand": "1000", "million": "1000000",
    }
    words = s.split()
    words = [number_words.get(w, w) for w in words]
    s = " ".join(words)

    # Normalize whitespace
    s = " ".join(s.split())

    return s


def exact_match(prediction: str, gold: str) -> float:
    """
    STRICT exact match after normalization. No partial containment.
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    return 1.0 if pred_norm == gold_norm else 0.0


def contains_match(prediction: str, gold: str) -> float:
    """
    Relaxed match: checks if gold is fully contained in prediction or vice versa,
    provided lengths are close.
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return 1.0

    if gold_norm and pred_norm:
        if gold_norm in pred_norm or pred_norm in gold_norm:
            pred_words = pred_norm.split()
            gold_words = gold_norm.split()
            if abs(len(pred_words) - len(gold_words)) <= 2:
                return 1.0

    return 0.0


def token_f1(prediction: str, gold: str) -> float:
    """
    Computes token-level F1 score after normalization.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, gold: str) -> float:
    """
    Computes ROUGE-L f-measure between prediction and gold.
    """
    if not HAS_ROUGE:
        return 0.0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(gold, prediction)
    return scores['rougeL'].fmeasure


def bert_score_metric(prediction: str, gold: str) -> Dict[str, float]:
    """
    Computes BERTScore precision, recall, and f1.
    """
    if not HAS_BERT_SCORE or not prediction.strip() or not gold.strip():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    P, R, F1 = bert_score([prediction], [gold], lang="en", verbose=False)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }


def supporting_doc_f1(retrieved_titles: List[str], gold_titles: List[str]) -> float:
    """
    Computes F1 score for retrieved supporting documents based on titles.
    """
    if not gold_titles:
        return 1.0
    matched = sum(
        1 for gt in gold_titles
        if any(gt.lower() in rt.lower() or rt.lower() in gt.lower()
               for rt in retrieved_titles)
    )
    recall = matched / len(gold_titles)
    precision = matched / len(retrieved_titles) if retrieved_titles else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def retrieval_recall_at_k(retrieved_titles: List[str], gold_titles: List[str], k: Optional[int] = None) -> float:
    """
    Computes Recall@K for document retrieval.
    """
    if not gold_titles:
        return 1.0
    
    if k is not None:
        retrieved_titles = retrieved_titles[:k]
        
    matched = sum(
        1 for gt in gold_titles
        if any(gt.lower() in rt.lower() or rt.lower() in gt.lower()
               for rt in retrieved_titles)
    )
    return matched / len(gold_titles)


def retrieval_mrr(retrieved_titles: List[str], gold_titles: List[str]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) for document retrieval.
    """
    if not gold_titles:
        return 1.0
    
    for rank, rt in enumerate(retrieved_titles, start=1):
        if any(gt.lower() in rt.lower() or rt.lower() in gt.lower() for gt in gold_titles):
            return 1.0 / rank
            
    return 0.0


def compute_aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Aggregates per-sample metrics into overall corpus metrics.
    """
    n = len(results)
    if n == 0:
        return {}
    
    em_scores = [r.get("em", 0) for r in results]
    f1_scores = [r.get("f1", 0) for r in results]
    contains_scores = [r.get("contains", 0) for r in results]
    rouge_l_scores = [r.get("rouge_l", 0) for r in results]
    
    metrics = {
        "n_samples": n,
        "em": sum(em_scores) / n,
        "f1": sum(f1_scores) / n,
        "contains_match": sum(contains_scores) / n,
        "rouge_l": sum(rouge_l_scores) / n,
    }
    
    if any("bert_score_f1" in r for r in results):
        metrics["bert_score_f1"] = sum(r.get("bert_score_f1", 0) for r in results) / n
        
    failures = [r for r in results if r.get("has_failure", False)]
    corrections = [r for r in results if r.get("correction_attempted", False)]
    succeeded = [r for r in results if r.get("correction_succeeded", False)]
    
    metrics["failure_detected_rate"] = len(failures) / n
    metrics["correction_attempted_rate"] = len(corrections) / len(failures) if failures else 0.0
    metrics["correction_success_rate"] = len(succeeded) / len(corrections) if corrections else 0.0

    sup_f1s = [r["sup_f1"] for r in results if r.get("sup_f1") is not None]
    if sup_f1s:
        metrics["supporting_facts_f1"] = sum(sup_f1s) / len(sup_f1s)
        
    recall_at_k = [r["recall_at_k"] for r in results if r.get("recall_at_k") is not None]
    if recall_at_k:
        metrics["retrieval_recall_at_k"] = sum(recall_at_k) / len(recall_at_k)
        
    mrr_scores = [r["mrr"] for r in results if r.get("mrr") is not None]
    if mrr_scores:
        metrics["retrieval_mrr"] = sum(mrr_scores) / len(mrr_scores)

    latencies = [r.get("latency_s", 0) for r in results if "latency_s" in r]
    if latencies:
        metrics["avg_latency_s"] = sum(latencies) / len(latencies)

    faithfulness = [r.get("faithfulness", 0) for r in results if "faithfulness" in r]
    if faithfulness:
        metrics["avg_faithfulness"] = sum(faithfulness) / len(faithfulness)

    return metrics


def compute_per_type_metrics(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Computes aggregate metrics broken down by question type (e.g., bridge, comparison).
    """
    by_type = {}
    for r in results:
        q_type = r.get("question_type", "unknown")
        if not q_type:
            q_type = "unknown"
        if q_type not in by_type:
            by_type[q_type] = []
        by_type[q_type].append(r)
        
    per_type_metrics = {}
    for q_type, type_results in by_type.items():
        per_type_metrics[q_type] = compute_aggregate_metrics(type_results)
        
    return per_type_metrics


def print_metrics(metrics: Dict, title: str = "Results"):
    """
    Prints metrics in a clean readable format.
    """
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<35}: {v:.4f}")
        else:
            print(f"  {k:<35}: {v}")
    print(f"{'='*50}\n")


def format_results_table(metrics_list: List[tuple]) -> str:
    """
    Formats a list of (name, metrics_dict) into a markdown-style table.
    """
    if not metrics_list:
        return ""
    headers = ["System", "EM", "F1", "ROUGE-L", "Corr. Success"]
    rows = []
    for name, m in metrics_list:
        rows.append([
            name,
            f"{m.get('em',0):.3f}",
            f"{m.get('f1',0):.3f}",
            f"{m.get('rouge_l',0):.3f}",
            f"{m.get('correction_success_rate',0):.3f}",
        ])
    col_widths = [max(len(h), max(len(r[i]) for r in rows))
                  for i, h in enumerate(headers)]
    sep = "+" + "+".join("-"*(w+2) for w in col_widths) + "+"
    header_row = "| " + " | ".join(h.ljust(w) for h,w in zip(headers,col_widths)) + " |"
    lines = [sep, header_row, sep]
    for row in rows:
        lines.append("| " + " | ".join(v.ljust(w) for v,w in zip(row,col_widths)) + " |")
    lines.append(sep)
    return "\n".join(lines)
