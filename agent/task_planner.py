"""
agent/task_planner.py — Decides how to attack a question.

Strategies
  direct      single retrieval + answer
  sequential  bridge question: answer sub-question 1, substitute its answer
              into sub-question 2 ("#1" placeholder), then answer that
  parallel    comparison question: answer each side independently, then compare
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_COMPARISON_CUES = ("taller", "older", "younger", "more", "less", "larger", "smaller", "first",
                    "earlier", "later", "longer", "shorter", "bigger", "higher", "lower", "same",
                    "both", "either")
_BRIDGE_CUES = ("that is", "who was", "who is", "whose", "which is", "born in", "directed by",
                "the director of", "the author of", "the founder of", "the city where",
                "the country where", "the band that", "the company that")


@dataclass
class PlanResult:
    """Output of the planner."""

    strategy: str
    sub_questions: List[str] = field(default_factory=list)
    original: str = ""

    # Dict-style access keeps older code (plan["strategy"]) working.
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class TaskPlanner:
    """Detects multi-hop structure and decomposes questions with the LLM."""

    MIN_SUB_QUESTION_LENGTH = 8

    def __init__(self, llm: Any, enable_decomposition: bool = True) -> None:
        self.llm = llm
        self.enable_decomposition = enable_decomposition

    def plan(self, question: str, question_type: Optional[str] = None) -> PlanResult:
        strategy = self.detect_strategy(question, question_type)
        if strategy == "direct" or not self.enable_decomposition:
            return PlanResult(strategy=strategy, sub_questions=[], original=question)

        if getattr(self.llm, "is_api", False):
            try:
                subs = self.llm.decompose(question, strategy)
            except Exception as err:
                logger.warning("Decomposition failed (%s); answering directly.", err)
                subs = []
            subs = [
                q for q in subs
                if len(q) > self.MIN_SUB_QUESTION_LENGTH and q.strip().lower() != question.strip().lower()
            ]
            if len(subs) >= 2:
                return PlanResult(strategy=strategy, sub_questions=subs[:2], original=question)

        logger.info("No usable decomposition; answering '%s' directly with full context.", strategy)
        return PlanResult(strategy=strategy, sub_questions=[], original=question)

    @staticmethod
    def detect_strategy(question: str, question_type: Optional[str] = None) -> str:
        """Map a dataset question type (or keyword cues) to a strategy."""
        if question_type:
            qt = question_type.lower()
            if qt == "bridge":
                return "sequential"
            if qt == "comparison":
                return "parallel"
            if qt in ("sequential", "parallel"):
                return qt
            return "direct"

        q = question.lower()
        if (" or " in q or " and " in q) and any(re.search(rf"\b{c}\b", q) for c in _COMPARISON_CUES):
            return "parallel"
        if any(cue in q for cue in _BRIDGE_CUES):
            return "sequential"
        return "direct"

    # Backwards-compatible name
    _detect_strategy = detect_strategy


def substitute_previous_answer(sub_question: str, previous_answer: str) -> str:
    """Replace '#1' (or a trailing '[answer]'-style placeholder) with the previous hop's answer."""
    if not previous_answer:
        return sub_question
    out = re.sub(r"#1\b", previous_answer, sub_question)
    out = re.sub(r"\[(answer|ans|a1|entity)[^\]]*\]", previous_answer, out, flags=re.I)
    return out
