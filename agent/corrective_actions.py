"""
agent/corrective_actions.py
Corrective actions — smarter application of recovery strategies.
"""
import re
import logging
from typing import List, Dict, Any, Optional

import config
from config import LLM_BACKEND

logger = logging.getLogger(__name__)

# Failure type → Strategy mapping
FAILURE_STRATEGY_MAP = {
    "empty": "expand_retrieve",
    "refusal": "reformulate", 
    "irrelevant": "re_retrieve",
    "ungrounded": "re_retrieve",
    "hallucinated": "expand_retrieve",
    "inconsistent": "decompose",
}


class CorrectiveActions:
    """Applies corrective actions when self-monitoring detects failures."""

    def __init__(self, llm: Any, retriever: Any, cost_tracker: Any = None) -> None:
        """
        Initialize CorrectiveActions.
        
        Args:
            llm: The LLM reasoner instance.
            retriever: The retrieval module instance.
            cost_tracker: Optional cost tracker instance for metric tracking.
        """
        self.llm = llm
        self.retriever = retriever
        self.cost_tracker = cost_tracker
        self._round = 0
        self._best_answer: Optional[str] = None
        self._previous_strategy: Optional[str] = None

    def apply(
        self, action: str, question: str, previous_answer: str,
        retrieved_docs: List[str], ungrounded_sentences: Optional[List[str]] = None,
        failure_type: str = ""
    ) -> Dict[str, Any]:
        """
        Apply a corrective action to recover from a failure adaptively based on failure type.
        
        Args:
            action: The suggested action from the monitor.
            question: The user query.
            previous_answer: The answer generated before the correction.
            retrieved_docs: Currently retrieved context documents.
            ungrounded_sentences: Sentences that failed grounding checks.
            failure_type: The categorized failure type detected by the self monitor.
            
        Returns:
            A dictionary containing new query, docs, answer, and action taken.
        """
        self._round += 1

        if self._best_answer is None:
            self._best_answer = previous_answer

        # Try mapping failure type to strategy first, fallback to monitor's action suggestion
        strategy = FAILURE_STRATEGY_MAP.get(failure_type, action)

        if strategy == self._previous_strategy:
            # Escalate if we already tried this strategy in the previous round
            if strategy in ("re_retrieve", "expand_retrieve"):
                strategy = "decompose"
            elif strategy == "decompose":
                strategy = "reformulate"
            else:
                strategy = "expand_retrieve"

        # Fallback to round-based escalation for unknown failure types
        if not strategy:
            if self._round == 1:
                strategy = "re_retrieve"
            elif self._round == 2:
                strategy = "decompose"
            else:
                strategy = "reformulate"

        self._previous_strategy = strategy

        if strategy == "re_retrieve" or strategy == "expand_retrieve":
            result = self._expand_retrieve(question, retrieved_docs)
        elif strategy == "decompose":
            result = self._decompose(question, retrieved_docs)
        else:
            result = self._reformulate(question, previous_answer, retrieved_docs)

        new_answer = result["new_answer"]
        if self._is_better(new_answer, previous_answer, question):
            result["new_answer"] = new_answer
            logger.info("Correction improved the answer.")
        else:
            result["new_answer"] = previous_answer
            result["action_taken"] += " [kept original — correction was worse]"
            logger.info("Correction worse. Kept original answer.")

        return result

    def reset_round(self) -> None:
        """Reset the internal state for a new query."""
        self._round = 0
        self._best_answer = None
        self._previous_strategy = None

    def _is_better(self, new_answer: str, old_answer: str, question: str) -> bool:
        """Heuristic to evaluate if the new answer is an improvement."""
        bad_phrases = ["i don't know", "cannot", "not mentioned",
                       "no information", "unclear"]
        new_lower = new_answer.lower().strip()
        old_lower = old_answer.lower().strip()

        if not new_lower or any(p in new_lower for p in bad_phrases):
            return False

        if len(old_lower.split()) <= 4 and not any(p in old_lower for p in bad_phrases):
            if len(new_lower.split()) > 6:
                return False

        return True

    def _expand_retrieve(self, question: str, retrieved_docs: List[str]) -> Dict[str, Any]:
        """Add more docs to the context with a varied query."""
        varied_query = f"more details about {question}"
        logger.info("Expanding retrieval with query: %s", varied_query)
        
        extra_raw = self.retriever.retrieve(varied_query)
        if self.cost_tracker:
            self.cost_tracker.record_retrieval_call()
        extra_docs = [d["text"] for d in extra_raw]
        
        max_docs = getattr(config, 'MAX_DOCS_PER_QUERY', 10)
        all_docs = list(dict.fromkeys(retrieved_docs + extra_docs))[:max_docs]
        
        new_answer = self.llm.answer_with_context(question, all_docs)
        if self.cost_tracker:
            self.cost_tracker.record_llm_call()

        return {
            "new_query": varied_query,
            "new_docs": all_docs,
            "new_answer": new_answer,
            "action_taken": f"expand_retrieve ({len(all_docs)} docs)",
        }

    def _decompose(self, question: str, retrieved_docs: List[str]) -> Dict[str, Any]:
        """Break into sub-questions and synthesize."""
        logger.info("Attempting decomposition for question: %s", question)
        if LLM_BACKEND in ("gemini", "groq", "openai", "ollama"):
            prompt = (
                f"Split into 2 simpler sub-questions.\n"
                f"Question: {question}\n"
                f"Output only 2 sub-questions numbered 1. and 2.:"
            )
            response = self.llm.generate(prompt, max_tokens=80)
            if self.cost_tracker:
                self.cost_tracker.record_llm_call()
            lines = [re.sub(r"^\d+[\.\)]\s*", "", l.strip())
                     for l in response.split("\n") if l.strip()]
            sub_qs = [l for l in lines if len(l) > 5][:2]
        else:
            sub_qs = self._heuristic_split(question)

        if not sub_qs:
            logger.info("Decomposition failed, falling back to expand_retrieve")
            return self._expand_retrieve(question, retrieved_docs)

        sub_answers = []
        all_docs = list(retrieved_docs)
        for sq in sub_qs:
            extra = self.retriever.retrieve(sq)
            if self.cost_tracker:
                self.cost_tracker.record_retrieval_call()
            combined = list(dict.fromkeys(all_docs + [d["text"] for d in extra]))[:8]
            sa = self.llm.answer_with_context(sq, combined)
            if self.cost_tracker:
                self.cost_tracker.record_llm_call()
            sub_answers.append(sa)
            all_docs = combined

        final = self.llm.answer_with_sub_answers(question, sub_answers, all_docs)
        if self.cost_tracker:
            self.cost_tracker.record_llm_call()

        return {
            "new_query": f"[decomposed] {sub_qs}",
            "new_docs": all_docs,
            "new_answer": final,
            "action_taken": f"decompose → {sub_qs}",
        }

    def _reformulate(self, question: str, previous_answer: str, retrieved_docs: List[str]) -> Dict[str, Any]:
        """Rephrase question as a last resort."""
        logger.info("Reformulating question: %s", question)
        if LLM_BACKEND in ("gemini", "groq", "openai", "ollama"):
            prompt = (
                f"Rephrase this question using different keywords.\n"
                f"Question: {question}\n"
                f"Rephrased (output only the question):"
            )
            new_query = self.llm.generate(prompt, max_tokens=40).strip().strip('"\'')
            if self.cost_tracker:
                self.cost_tracker.record_llm_call()
            if len(new_query) < 5:
                new_query = question
        else:
            words = question.rstrip("?").split()
            new_query = " ".join(words[-3:] + words[:3]) if len(words) > 4 else question

        new_docs_raw = self.retriever.retrieve(new_query)
        if self.cost_tracker:
            self.cost_tracker.record_retrieval_call()
        new_docs = [d["text"] for d in new_docs_raw]
        all_docs = list(dict.fromkeys(retrieved_docs + new_docs))[:8]
        new_answer = self.llm.answer_with_context(question, all_docs)
        if self.cost_tracker:
            self.cost_tracker.record_llm_call()

        return {
            "new_query": new_query,
            "new_docs": all_docs,
            "new_answer": new_answer,
            "action_taken": f"reformulate → '{new_query[:50]}'",
        }

    def _heuristic_split(self, question: str) -> List[str]:
        """Fallback split logic using common bridge words."""
        q = question.lower()
        for pivot in ["that", "who", "which", "where", "whose"]:
            if f" {pivot} " in q:
                idx = q.index(f" {pivot} ")
                left = question[:idx].strip().rstrip(",")
                right = question[idx + 1:].strip()
                if len(left) > 5 and len(right) > 5:
                    return [left + "?", right + "?"]
        words = question.split()
        mid = len(words) // 2
        return [" ".join(words[:mid]) + "?", " ".join(words[mid:]) + "?"]
