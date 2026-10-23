"""
agent/corrective_actions.py — Self-healing strategies.

The monitor's failure type picks the first strategy; if a strategy was
already tried for this question the next untried one is used, so the loop
never repeats itself.

  re_retrieve      LLM writes a targeted search query for the *missing* evidence;
                   passages already in context are excluded so new evidence surfaces
  expand_retrieve  widen the context with passages about the question AND the
                   current answer/bridge entity
  decompose        split into sub-questions (with #1 substitution) and synthesise
  reformulate      rephrase the question, retrieve with it, answer the original
  adopt_last_hop   (inconsistent multi-hop synthesis) use the verified last-hop answer; 0 LLM calls

Retrieval-based strategies are skipped when every passage is already in the
context (nothing new can be found), which saves LLM calls on small corpora.

Choosing *which* candidate answer to keep is the pipeline's job: it re-runs
the monitor on every candidate and keeps the best-grounded one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import config
from agent.cost_tracker import CURRENT_TRACKER
from agent.multihop import merge_docs, run_subquestions
from agent.self_monitor import FAILURE_STRATEGY_MAP, is_refusal
from agent.task_planner import TaskPlanner

logger = logging.getLogger(__name__)

ESCALATION_ORDER = ("re_retrieve", "decompose", "expand_retrieve", "reformulate")
RETRIEVAL_STRATEGIES = {"re_retrieve", "expand_retrieve", "reformulate"}


@dataclass
class CorrectionResult:
    strategy: str
    new_answer: str
    new_docs: List[str]
    new_query: str = ""
    action_taken: str = ""
    sub_qas: List[Tuple[str, str]] = field(default_factory=list)
    plan_strategy: str = "direct"  # "sequential"/"parallel" when the answer came from sub-questions

    # Dict-style access for backwards compatibility (result["new_answer"]).
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy, "new_answer": self.new_answer, "new_query": self.new_query,
            "action_taken": self.action_taken, "n_docs": len(self.new_docs), "sub_qas": self.sub_qas,
        }


class CorrectiveActions:
    """Applies corrective strategies when the monitor flags a failure."""

    def __init__(
        self,
        llm: Any,
        retriever: Any = None,
        cost_tracker: Any = None,
        top_k: Optional[int] = None,
        max_docs: Optional[int] = None,
        short_answers: bool = True,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.cost_tracker = cost_tracker
        self.top_k = top_k
        self.max_docs = max_docs
        self.short_answers = short_answers
        self._tried: List[str] = []

    def reset_round(self) -> None:
        self._tried = []

    def choose_strategy(self, failure_type: str, suggested: str = "", can_retrieve: bool = True,
                        has_sub_answers: bool = False) -> Optional[str]:
        """
        Pick the strategy for this failure: the mapped one if it is untried and
        feasible, otherwise the next untried feasible one. None = nothing left to try.
        """
        def feasible(s: str) -> bool:
            if s in self._tried:
                return False
            if s in RETRIEVAL_STRATEGIES and not can_retrieve:
                return False
            return s != "adopt_last_hop" or has_sub_answers

        preferred = FAILURE_STRATEGY_MAP.get(failure_type) or suggested or "re_retrieve"
        if preferred not in (*ESCALATION_ORDER, "adopt_last_hop"):
            preferred = "re_retrieve"
        if feasible(preferred):
            return preferred
        return next((s for s in ESCALATION_ORDER if feasible(s)), None)

    def apply(
        self,
        action: str,
        question: str,
        previous_answer: str,
        retrieved_docs: Sequence[str],
        ungrounded_sentences: Optional[List[str]] = None,
        failure_type: str = "",
        index: Any = None,
        question_type: Optional[str] = None,
        sub_qas: Optional[List[Tuple[str, str]]] = None,
    ) -> Optional[CorrectionResult]:
        """Apply the best remaining strategy. Returns None when no strategy can help any more."""
        top_k = self.top_k or config.TOP_K_RETRIEVAL
        max_docs = self.max_docs or config.MAX_DOCS_PER_QUERY
        docs = list(retrieved_docs)
        can_retrieve = index is None or len(index) > len(set(docs))
        strategy = self.choose_strategy(failure_type, action, can_retrieve=can_retrieve,
                                        has_sub_answers=bool(sub_qas))
        if strategy is None:
            logger.info("No untried corrective strategy left (failure_type=%s).", failure_type)
            return None
        self._tried.append(strategy)
        logger.info("Correction strategy: %s (failure_type=%s)", strategy, failure_type or "?")

        if strategy == "adopt_last_hop":
            last_q, last_a = sub_qas[-1]
            result = CorrectionResult("adopt_last_hop", last_a, docs, last_q,
                                      f"adopt_last_hop: final answer disagreed with hop '{last_q[:50]}'",
                                      list(sub_qas), "sequential")
        elif strategy == "decompose":
            result = self._decompose(question, docs, index, question_type, top_k, max_docs)
        elif strategy == "expand_retrieve":
            result = self._expand(question, previous_answer, docs, index, top_k, max_docs)
        elif strategy == "reformulate":
            result = self._reformulate(question, docs, index, top_k, max_docs)
        else:
            result = self._targeted(question, previous_answer, docs, index, ungrounded_sentences or [],
                                    top_k, max_docs)
        tracker = self.cost_tracker or CURRENT_TRACKER.get()
        if tracker:
            tracker.record_correction_round()
        return result

    # ── Strategies ───────────────────────────────────────────────────────────
    def _answer(self, question: str, docs: Sequence[str]) -> str:
        return self.llm.answer_with_context(question, docs, short=self.short_answers)

    def _search(self, index: Any, query: str, top_k: int, exclude: Sequence[str] = ()) -> List[str]:
        if index is None:
            return []
        tracker = self.cost_tracker or CURRENT_TRACKER.get()
        if tracker:
            tracker.record_retrieval_call()
        return index.search_texts(query, top_k=top_k, exclude=list(exclude))

    def _targeted(self, question, previous_answer, docs, index, ungrounded, top_k, max_docs) -> CorrectionResult:
        if getattr(self.llm, "is_api", False):
            query = self.llm.search_query(question, previous_answer, ungrounded)
        else:
            bridge = "" if is_refusal(previous_answer) else previous_answer
            query = f"{question} {bridge}".strip()
        new = self._search(index, query, top_k, exclude=docs)
        # Keep the existing evidence and append the new passages (bounded growth).
        all_docs = merge_docs(docs, new, max(max_docs, len(docs)) + top_k)
        return CorrectionResult("re_retrieve", self._answer(question, all_docs), all_docs, query,
                                f"re_retrieve: '{query[:60]}' (+{len(new)} new docs)")

    def _expand(self, question, previous_answer, docs, index, top_k, max_docs) -> CorrectionResult:
        extra = self._search(index, question, top_k * 2, exclude=docs)
        if previous_answer and not is_refusal(previous_answer):
            extra = merge_docs(extra, self._search(index, previous_answer, 2, exclude=docs), 10 ** 6)
        all_docs = merge_docs(docs, extra, max(max_docs, len(docs)) + top_k)
        return CorrectionResult("expand_retrieve", self._answer(question, all_docs), all_docs, question,
                                f"expand_retrieve: {len(docs)} -> {len(all_docs)} docs")

    def _decompose(self, question, docs, index, question_type, top_k, max_docs) -> CorrectionResult:
        strategy = TaskPlanner.detect_strategy(question, question_type)
        strategy = "sequential" if strategy == "direct" else strategy
        subs: List[str] = []
        if getattr(self.llm, "is_api", False):
            try:
                subs = self.llm.decompose(question, strategy)
            except Exception as err:
                logger.warning("Decomposition call failed: %s", err)
        else:
            subs = self._heuristic_split(question)
        if len(subs) < 2:
            logger.info("Decomposition unusable; falling back to expand_retrieve.")
            return self._expand(question, "", docs, index, top_k, max_docs)
        final, all_docs, sub_qas = run_subquestions(
            self.llm, index, question, subs[:2], strategy, docs, top_k, max_docs, short=self.short_answers
        )
        return CorrectionResult("decompose", final, all_docs, " | ".join(q for q, _ in sub_qas),
                                f"decompose ({strategy}): {[q for q, _ in sub_qas]}", sub_qas, strategy)

    def _reformulate(self, question, docs, index, top_k, max_docs) -> CorrectionResult:
        if getattr(self.llm, "is_api", False):
            new_q = self.llm.rephrase(question)
        else:
            words = question.rstrip("?").split()
            new_q = " ".join(words[-3:] + words[:3]) if len(words) > 4 else question
        new = self._search(index, new_q, top_k, exclude=docs)
        all_docs = merge_docs(docs, new, max(max_docs, len(docs)) + top_k)
        return CorrectionResult("reformulate", self._answer(question, all_docs), all_docs, new_q,
                                f"reformulate: '{new_q[:60]}'")

    @staticmethod
    def _heuristic_split(question: str) -> List[str]:
        """Fallback split on relative pronouns (used without an API model)."""
        q = question.lower()
        for pivot in ("that", "who", "which", "where", "whose"):
            token = f" {pivot} "
            if token in q:
                idx = q.index(token)
                left, right = question[:idx].strip().rstrip(","), question[idx + 1:].strip()
                if len(left) > 5 and len(right) > 5:
                    return [left.rstrip("?") + "?", right.rstrip("?") + "?"]
        words = question.split()
        mid = len(words) // 2
        return [" ".join(words[:mid]) + "?", " ".join(words[mid:])]
