"""
agent/self_monitor.py — Self-Monitoring Module

KEY INSIGHT from results analysis:
The monitor was flagging TOO MANY answers (48%) and corrections
were making things worse. We need to be MORE SELECTIVE — only flag
answers we're highly confident are wrong.

New strategy:
  - Only flag clear uncertainty phrases (Check 1)
  - Only flag answers completely absent from docs (Check 2, stricter)
  - Only flag empty/garbage answers (Check 3)
  - LLM verify ONLY when other checks pass AND answer seems suspicious
  - Skip LLM verify for short confident answers (saves API calls too)
"""
import re
from dataclasses import dataclass, field
from config import LLM_BACKEND

UNCERTAINTY_PHRASES = [
    "i don't know", "i do not know", "cannot be determined",
    "not mentioned", "not provided", "no information",
    "i'm not sure", "not specified", "not available",
    "not enough information", "cannot answer", "cannot find",
    "the context does not", "there is no information",
    "based on the context, i cannot",
]

STOPWORDS = {
    "the","a","an","is","was","are","were","be","been","it","its","of",
    "in","on","at","to","for","and","or","but","by","with","from","that",
    "this","which","what","who","where","when","how","did","does","do",
    "has","have","had","will","would","could","should","may","might",
    "their","our","your","my","his","her","they","we","he","she","i",
    "yes","no","after","before","during","between","both","either",
}


@dataclass
class FailureReport:
    has_failure: bool = False
    confidence_ok: bool = True
    grounding_ok: bool = True
    completeness_ok: bool = True
    llm_verify_ok: bool = True
    consistency_ok: bool = True
    low_confidence: bool = False
    ungrounded_sentences: list = field(default_factory=list)
    failure_reason: str = ""
    suggested_action: str = ""


class SelfMonitor:
    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, question, answer, retrieved_docs, sub_answers=None):
        report = FailureReport()

        # Check 1: Confidence — only flag clear uncertainty phrases
        report.confidence_ok, report.low_confidence = self._check_confidence(answer)

        # Check 2: Completeness — flag empty/garbage only
        report.completeness_ok = self._check_completeness(answer)

        # Check 3: Grounding — only flag if answer has ZERO overlap with docs
        # (much stricter than before — was causing too many false positives)
        if report.confidence_ok and report.completeness_ok:
            report.grounding_ok, report.ungrounded_sentences = \
                self._check_grounding_strict(answer, retrieved_docs)

        # Check 4: LLM verify — ONLY for suspicious cases
        # Skip for: short answers, yes/no, numbers (these are usually correct)
        if (report.confidence_ok and report.completeness_ok
                and report.grounding_ok
                and LLM_BACKEND in ("gemini", "groq", "openai", "ollama")
                and self._is_suspicious(answer, question)):
            report.llm_verify_ok = self._llm_verify(question, answer, retrieved_docs)

        # Multi-hop consistency
        if sub_answers and len(sub_answers) >= 2:
            report.consistency_ok = self._check_consistency(sub_answers)

        if not all([report.confidence_ok, report.grounding_ok,
                    report.completeness_ok, report.llm_verify_ok,
                    report.consistency_ok]):
            report.has_failure = True
            report.failure_reason = self._build_reason(report)
            report.suggested_action = self._suggest_action(report)

        return report

    def _check_confidence(self, answer):
        a = answer.lower().strip().rstrip(".")
        for phrase in UNCERTAINTY_PHRASES:
            if phrase in a:
                return False, True
        if not a:
            return False, True
        return True, False

    def _check_completeness(self, answer):
        a = answer.lower().strip()
        if not a or a in (".", ",", "?", "-", "n/a", "none", ""):
            return False
        # Only flag as incomplete if it starts with uncertainty
        for phrase in UNCERTAINTY_PHRASES:
            if a.startswith(phrase):
                return False
        return True

    def _check_grounding_strict(self, answer, retrieved_docs):
        """
        STRICT grounding: only flag if answer has ZERO meaningful
        word overlap with any retrieved document.
        This prevents false positives that cause corrections to make
        things worse.
        """
        if not retrieved_docs or not answer.strip():
            return True, []

        answer_words = [
            w.lower().strip(".,!?\"'()-") for w in answer.split()
            if len(w) > 3 and w.lower().strip(".,") not in STOPWORDS
        ]

        # Very short answers (1-2 meaningful words): skip grounding check
        # These are usually correct entity names
        if len(answer_words) <= 1:
            return True, []

        combined = " ".join(retrieved_docs).lower()
        matched  = sum(1 for w in answer_words if w in combined)

        # Only flag if ZERO words match — truly ungrounded
        if matched == 0 and len(answer_words) >= 2:
            return False, [answer]
        return True, []

    def _is_suspicious(self, answer, question):
        """
        Only run LLM verify for answers that look suspicious.
        Skip for: yes/no, numbers, single common words.
        This reduces unnecessary API calls and false positives.
        """
        a = answer.lower().strip()
        # Skip yes/no — these are almost always right when model says them
        if a in ("yes", "no", "true", "false"):
            return False
        # Skip pure numbers — usually correct
        if re.match(r"^\d+$", a):
            return False
        # Skip very short confident answers (1-2 words)
        if len(answer.split()) <= 2:
            return False
        # Only verify longer answers that might be hallucinated
        return True

    def _llm_verify(self, question, answer, retrieved_docs):
        """LLM verification — only called for suspicious longer answers."""
        context = " ".join(d[:200] for d in retrieved_docs[:3])
        prompt  = (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Context: {context}\n\n"
            f"Is this answer supported by the context? Reply YES or NO only."
        )
        try:
            response = self.llm.generate(prompt, max_tokens=5).strip().upper()
            return not response.startswith("NO")
        except Exception:
            return True  # on error, don't flag

    def _check_consistency(self, sub_answers):
        lowers  = [a.lower().strip().rstrip(".") for a in sub_answers]
        has_yes = any(a == "yes" or a.startswith("yes,") for a in lowers)
        has_no  = any(a == "no"  or a.startswith("no,")  for a in lowers)
        return not (has_yes and has_no)

    def _build_reason(self, report):
        parts = []
        if not report.confidence_ok:    parts.append("uncertain answer")
        if not report.grounding_ok:     parts.append("answer not in docs")
        if not report.completeness_ok:  parts.append("empty/missing answer")
        if not report.llm_verify_ok:    parts.append("LLM verification failed")
        if not report.consistency_ok:   parts.append("sub-answers contradict")
        return "; ".join(parts)

    def _suggest_action(self, report):
        if not report.completeness_ok:  return "decompose"
        if not report.grounding_ok:     return "re_retrieve"
        if not report.llm_verify_ok:    return "re_retrieve"
        if not report.confidence_ok:    return "reformulate"
        return "reformulate"
