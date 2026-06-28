"""
agent/corrective_actions.py
Corrective actions — now smarter about WHEN to apply each strategy.

KEY FIX: Previous corrections were replacing correct answers with wrong ones.
New approach:
  - Round 1: Re-retrieve with EXPANDED context (add more docs, don't replace)
  - Round 2: Try decomposition for multi-hop
  - Round 3: Reformulate only as last resort
  
Also: if the new answer is shorter/worse than the original, KEEP the original.
"""
import re
from config import LLM_BACKEND
from evaluation.metrics import token_f1


class CorrectiveActions:
    def __init__(self, llm, retriever):
        self.llm       = llm
        self.retriever = retriever
        self._round    = 0
        self._best_answer = None
        self._best_score  = -1

    def apply(self, action, question, previous_answer,
              retrieved_docs, ungrounded_sentences=None):
        self._round += 1

        # Track best answer seen so far
        if self._best_answer is None:
            self._best_answer = previous_answer

        # Round-based escalation
        if self._round == 1:
            result = self._expand_retrieve(question, retrieved_docs)
        elif self._round == 2:
            result = self._decompose(question, retrieved_docs)
        else:
            result = self._reformulate(question, previous_answer, retrieved_docs)

        # KEY FIX: Only use new answer if it's better than previous
        # Measure quality by answer length appropriateness
        new_answer = result["new_answer"]
        if self._is_better(new_answer, previous_answer, question):
            result["new_answer"] = new_answer
        else:
            # Keep previous answer if correction made things worse
            result["new_answer"] = previous_answer
            result["action_taken"] += " [kept original — correction was worse]"

        return result

    def reset_round(self):
        self._round       = 0
        self._best_answer = None
        self._best_score  = -1

    def _is_better(self, new_answer, old_answer, question):
        """
        Heuristic: new answer is better if:
        - It's not empty or "I don't know"
        - It's not significantly longer than old (verbose = worse)
        - It doesn't start with uncertainty phrases
        """
        bad_phrases = ["i don't know", "cannot", "not mentioned",
                       "no information", "unclear"]
        new_lower = new_answer.lower().strip()
        old_lower = old_answer.lower().strip()

        # If new is empty or uncertain — keep old
        if not new_lower or any(p in new_lower for p in bad_phrases):
            return False

        # If old was already good (short, confident) — be careful
        if len(old_lower.split()) <= 4 and not any(
                p in old_lower for p in bad_phrases):
            # Only switch if new is also short and different
            if len(new_lower.split()) > 6:
                return False  # new is too verbose

        return True

    # ── Strategy 1: Expand Retrieval ──────────────────────────────────────────

    def _expand_retrieve(self, question, retrieved_docs):
        """Add MORE docs to context rather than replacing existing ones."""
        # Retrieve with different query — add to existing, don't replace
        extra_raw  = self.retriever.retrieve(question)
        extra_docs = [d["text"] for d in extra_raw]
        # Merge: existing + new, deduplicated, keep more
        all_docs   = list(dict.fromkeys(retrieved_docs + extra_docs))[:10]
        new_answer = self.llm.answer_with_context(question, all_docs)

        return {
            "new_query":    question,
            "new_docs":     all_docs,
            "new_answer":   new_answer,
            "action_taken": f"expand_retrieve ({len(all_docs)} docs)",
        }

    # ── Strategy 2: Decompose ─────────────────────────────────────────────────

    def _decompose(self, question, retrieved_docs):
        """Break into sub-questions and synthesize."""
        if LLM_BACKEND in ("gemini", "groq", "openai", "ollama"):
            prompt = (
                f"Split into 2 simpler sub-questions.\n"
                f"Question: {question}\n"
                f"Output only 2 sub-questions numbered 1. and 2.:"
            )
            response = self.llm.generate(prompt, max_tokens=80)
            lines    = [re.sub(r"^\d+[\.\)]\s*", "", l.strip())
                        for l in response.split("\n") if l.strip()]
            sub_qs   = [l for l in lines if len(l) > 5][:2]
        else:
            sub_qs = self._heuristic_split(question)

        if not sub_qs:
            return self._expand_retrieve(question, retrieved_docs)

        sub_answers = []
        all_docs    = list(retrieved_docs)
        for sq in sub_qs:
            extra    = self.retriever.retrieve(sq)
            combined = list(dict.fromkeys(
                all_docs + [d["text"] for d in extra]))[:8]
            sa       = self.llm.answer_with_context(sq, combined)
            sub_answers.append(sa)
            all_docs = combined

        final = self.llm.answer_with_sub_answers(question, sub_answers, all_docs)

        return {
            "new_query":    f"[decomposed] {sub_qs}",
            "new_docs":     all_docs,
            "new_answer":   final,
            "action_taken": f"decompose → {sub_qs}",
        }

    # ── Strategy 3: Reformulate ───────────────────────────────────────────────

    def _reformulate(self, question, previous_answer, retrieved_docs):
        """Rephrase question as last resort."""
        if LLM_BACKEND in ("gemini", "groq", "openai", "ollama"):
            prompt = (
                f"Rephrase this question using different keywords.\n"
                f"Question: {question}\n"
                f"Rephrased (output only the question):"
            )
            new_query = self.llm.generate(
                prompt, max_tokens=40).strip().strip('"\'')
            if len(new_query) < 5:
                new_query = question
        else:
            words     = question.rstrip("?").split()
            new_query = " ".join(
                words[-3:] + words[:3]) if len(words) > 4 else question

        new_docs_raw = self.retriever.retrieve(new_query)
        new_docs     = [d["text"] for d in new_docs_raw]
        all_docs     = list(dict.fromkeys(retrieved_docs + new_docs))[:8]
        new_answer   = self.llm.answer_with_context(question, all_docs)

        return {
            "new_query":    new_query,
            "new_docs":     all_docs,
            "new_answer":   new_answer,
            "action_taken": f"reformulate → '{new_query[:50]}'",
        }

    def _heuristic_split(self, question):
        q = question.lower()
        for pivot in ["that", "who", "which", "where", "whose"]:
            if f" {pivot} " in q:
                idx   = q.index(f" {pivot} ")
                left  = question[:idx].strip().rstrip(",")
                right = question[idx + 1:].strip()
                if len(left) > 5 and len(right) > 5:
                    return [left + "?", right + "?"]
        words = question.split()
        mid   = len(words) // 2
        return [" ".join(words[:mid]) + "?", " ".join(words[mid:]) + "?"]
