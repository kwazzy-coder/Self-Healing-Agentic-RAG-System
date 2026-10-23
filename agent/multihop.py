"""
agent/multihop.py — Executes a decomposition plan (shared by the planner path
and the 'decompose' corrective action).

For *sequential* (bridge) questions the answer of hop 1 is substituted into
hop 2 before retrieving for it, e.g.

    1. Who directed Inception?            -> Christopher Nolan
    2. What nationality is #1?            -> "What nationality is Christopher Nolan?"

The original code answered both sub-questions independently, so hop 2 was
retrieved with an unresolved placeholder and rarely found the right passage.
"""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from agent.task_planner import substitute_previous_answer


def merge_docs(primary: Sequence[str], secondary: Sequence[str], limit: int) -> List[str]:
    """Order-preserving de-duplicated union, primary first, capped at `limit`."""
    return list(dict.fromkeys([*primary, *secondary]))[:limit]


def run_subquestions(
    llm: Any,
    index: Any,
    question: str,
    sub_questions: Sequence[str],
    strategy: str,
    base_docs: Sequence[str],
    top_k: int,
    max_docs: int,
    short: bool = True,
) -> Tuple[str, List[str], List[Tuple[str, str]]]:
    """Answer each sub-question with its own retrieval, then synthesise.

    Returns (final_answer, accumulated_docs, [(sub_question, sub_answer), ...]).
    """
    docs = list(base_docs)
    sub_qas: List[Tuple[str, str]] = []
    previous = ""
    for i, sq in enumerate(sub_questions):
        if strategy == "sequential" and i > 0:
            sq = substitute_previous_answer(sq, previous)
        hits = index.search_texts(sq, top_k=top_k) if index is not None else []
        context = merge_docs(hits, docs, max_docs)
        answer = llm.answer_with_context(sq, context, short=True)
        sub_qas.append((sq, answer))
        docs = merge_docs(docs, hits, max_docs)
        previous = answer
    final = llm.answer_with_sub_answers(question, sub_qas, docs, short=short)
    return final, docs, sub_qas
