"""
agent/task_planner.py — Task planning with LLM decomposition enabled for Groq.
"""
import re
from config import LLM_BACKEND


class TaskPlanner:
    def __init__(self, llm):
        self.llm = llm

    def plan(self, question, question_type=None):
        strategy = self._detect_strategy(question_type)

        if strategy == "direct":
            return {"strategy": "direct", "sub_questions": [], "original": question}

        # Enable decomposition for all capable backends
        if LLM_BACKEND in ("groq", "openai", "ollama"):
            sub_questions = self._decompose_llm(question, strategy)
            # Validate sub-questions — reject garbage output
            sub_questions = [
                q for q in sub_questions
                if len(q) > 8
                and q.lower() != question.lower()
                and not q[0].isdigit()
            ]
            if len(sub_questions) >= 2:
                return {"strategy": strategy, "sub_questions": sub_questions, "original": question}

        # Fallback: direct with all docs
        return {"strategy": strategy, "sub_questions": [], "original": question}

    def _detect_strategy(self, question_type):
        if question_type == "bridge":     return "sequential"
        if question_type == "comparison": return "parallel"
        return "direct"

    def _decompose_llm(self, question, strategy):
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
        # Re-add the "1." we used as prompt continuation
        full = "1." + response if not response.startswith("1") else response
        lines = [l.strip() for l in full.split("\n") if l.strip()]
        sub_qs = []
        for line in lines:
            clean = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if clean:
                sub_qs.append(clean)
        return sub_qs[:2]
