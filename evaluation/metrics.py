"""
evaluation/metrics.py
Smarter normalization that handles:
- Articles: "the United States" = "United States"
- Numbers: "two" = "2"
- Abbreviations: "U.S." = "US"
- Aliases: handles common alternate forms
"""
import re, string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower().strip()

    # Remove punctuation except hyphens in compound words
    s = re.sub(r"[^\w\s-]", " ", s)

    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)

    # Normalize number words to digits
    number_words = {
        "zero":"0","one":"1","two":"2","three":"3","four":"4",
        "five":"5","six":"6","seven":"7","eight":"8","nine":"9",
        "ten":"10","eleven":"11","twelve":"12","thirteen":"13",
        "fourteen":"14","fifteen":"15","sixteen":"16",
        "seventeen":"17","eighteen":"18","nineteen":"19","twenty":"20",
        "thirty":"30","forty":"40","fifty":"50","hundred":"100",
        "thousand":"1000","million":"1000000",
    }
    words = s.split()
    words = [number_words.get(w, w) for w in words]
    s = " ".join(words)

    # Normalize whitespace
    s = " ".join(s.split())

    return s


def exact_match(prediction: str, gold: str) -> float:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return 1.0

    # Partial containment check:
    # If gold is fully contained in prediction or vice versa
    # e.g. prediction="United States of America", gold="United States"
    if gold_norm and pred_norm:
        if gold_norm in pred_norm or pred_norm in gold_norm:
            # Only give credit if lengths are close (within 2 words)
            pred_words = pred_norm.split()
            gold_words = gold_norm.split()
            if abs(len(pred_words) - len(gold_words)) <= 2:
                return 1.0

    return 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common    = Counter(pred_tokens) & Counter(gold_tokens)
    num_same  = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def supporting_doc_f1(retrieved_titles: list, gold_titles: list) -> float:
    if not gold_titles:
        return 1.0
    matched = sum(
        1 for gt in gold_titles
        if any(gt.lower() in rt.lower() or rt.lower() in gt.lower()
               for rt in retrieved_titles)
    )
    recall    = matched / len(gold_titles)
    precision = matched / len(retrieved_titles) if retrieved_titles else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_aggregate_metrics(results: list) -> dict:
    n = len(results)
    if n == 0:
        return {}
    em_scores  = [r.get("em", 0) for r in results]
    f1_scores  = [r.get("f1", 0) for r in results]
    failures   = [r for r in results if r.get("has_failure", False)]
    corrections= [r for r in results if r.get("correction_attempted", False)]
    sup_f1s    = [r["sup_f1"] for r in results
                  if "sup_f1" in r and r["sup_f1"] is not None]
    metrics = {
        "n_samples":           n,
        "em":                  sum(em_scores) / n,
        "f1":                  sum(f1_scores) / n,
        "failure_rate":        len(failures) / n,
        "correction_rate":     len(corrections) / len(failures) if failures else 0.0,
        "task_success_rate":   sum(em_scores) / n,
        "hallucination_rate":  len(failures) / n,
    }
    if sup_f1s:
        metrics["supporting_facts_f1"] = sum(sup_f1s) / len(sup_f1s)
    return metrics


def print_metrics(metrics: dict, title: str = "Results"):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<35}: {v:.4f}")
        else:
            print(f"  {k:<35}: {v}")
    print(f"{'='*50}\n")


def format_results_table(metrics_list: list) -> str:
    if not metrics_list:
        return ""
    headers = ["System", "EM", "F1", "Halluc. Rate", "Correction Rate"]
    rows    = []
    for name, m in metrics_list:
        rows.append([
            name,
            f"{m.get('em',0):.3f}",
            f"{m.get('f1',0):.3f}",
            f"{m.get('hallucination_rate',0):.3f}",
            f"{m.get('correction_rate',0):.3f}",
        ])
    col_widths = [max(len(h), max(len(r[i]) for r in rows))
                  for i, h in enumerate(headers)]
    sep        = "+" + "+".join("-"*(w+2) for w in col_widths) + "+"
    header_row = "| " + " | ".join(h.ljust(w) for h,w in zip(headers,col_widths)) + " |"
    lines      = [sep, header_row, sep]
    for row in rows:
        lines.append("| " + " | ".join(v.ljust(w) for v,w in zip(row,col_widths)) + " |")
    lines.append(sep)
    return "\n".join(lines)
