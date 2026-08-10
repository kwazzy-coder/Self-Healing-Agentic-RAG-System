"""
agent/task_planner.py — Task planning with LLM decomposition.
"""
import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Any
from config import LLM_BACKEND

logger = logging.getLogger(__name__)

@dataclass
class PlanResult:
    """Dataclass holding the result of task planning."""
    strategy: str
    sub_questions: List[str]
    original: str

class TaskPlanner:
    """Task planning with LLM decomposition."""
    
    MIN_SUB_QUESTION_LENGTH = 8
    
    def __init__(self, llm: Any) -> None:
        """
        Initialize the TaskPlanner.
        
        Args:
            llm: The LLM reasoner instance.
        """
        self.llm = llm

    def plan(self, question: str, question_type: Optional[str] = None) -> PlanResult:
        """
        Plan the task strategy and optionally decompose the question.
        
        Args:
            question: The original user question.
            question_type: Optional predefined question type.
            
        Returns:
            PlanResult containing the strategy, sub-questions, and original question.
        """
        strategy = self._detect_strategy(question, question_type)

        if strategy == "direct":
            logger.info("Using direct strategy for question: %s", question)
            return PlanResult(strategy="direct", sub_questions=[], original=question)

        # Enable decomposition for all capable backends
        if LLM_BACKEND in ("gemini", "groq", "openai", "ollama"):
            logger.info("Attempting LLM decomposition for strategy: %s", strategy)
            sub_questions = self._decompose_llm(question, strategy)
            sub_questions = [
                q for q in sub_questions
                if len(q) > self.MIN_SUB_QUESTION_LENGTH
                and q.lower() != question.lower()
                and not q[0].isdigit()
            ]
            if len(sub_questions) >= 2:
                logger.info("Successfully decomposed into %d sub-questions", len(sub_questions))
                return PlanResult(strategy=strategy, sub_questions=sub_questions, original=question)

        logger.info("Fallback to direct strategy for question: %s", question)
        return PlanResult(strategy=strategy, sub_questions=[], original=question)

    def _detect_strategy(self, question: str, question_type: Optional[str]) -> str:
        """
        Detect the reasoning strategy for a given question.
        
        Args:
            question: The question text.
            question_type: The explicitly provided question type, if any.
            
        Returns:
            The strategy string ('direct', 'sequential', or 'parallel').
        """
        if question_type:
            if question_type == "bridge":
                return "sequential"
            if question_type == "comparison":
                return "parallel"
            return "direct"
            
        # Auto-detect from question
        q_lower = question.lower()
        
        # Comparison patterns
        comparison_words = ["which", "who is", "taller", "older", "more", "less", "better", "worse", "larger", "smaller"]
        if any(word in q_lower for word in comparison_words) and (" or " in q_lower or " and " in q_lower):
            return "parallel"
            
        # Bridge patterns (relative clauses)
        bridge_words = ["that is", "who was", "whose", "which is", "born in", "directed by"]
        if any(word in q_lower for word in bridge_words):
            return "sequential"
            
        return "direct"

    def _decompose_llm(self, question: str, strategy: str) -> List[str]:
        """
        Decompose a question into sub-questions using the LLM.
        
        Args:
            question: The question to decompose.
            strategy: The strategy to use ('sequential' or 'parallel').
            
        Returns:
            A list of sub-questions.
        """
        if strategy == "sequential":
            prompt = (
                f"Break this 2-hop question into 2 sub-questions to answer in order.\n"
                f"The answer to sub-question 1 should help answer sub-question 2.\n"
                f"Output ONLY 2 sub-questions, numbered 1. and 2. No other text.\n\n"
                f"Question: {question}\n\n"
                f"1."
            )
        else:
            prompt = (
                f"Break this comparison question into 2 independent sub-questions.\n"
                f"Output ONLY 2 sub-questions, numbered 1. and 2. No other text.\n\n"
                f"Question: {question}\n\n"
                f"1."
            )

        response = self.llm.generate(prompt, max_tokens=80)
        full = "1." + response if not response.startswith("1") else response
        lines = [l.strip() for l in full.split("\n") if l.strip()]
        sub_qs = []
        for line in lines:
            clean = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if clean:
                sub_qs.append(clean)
        return sub_qs[:2]
