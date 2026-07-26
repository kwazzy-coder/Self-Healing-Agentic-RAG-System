"""
agent/self_monitor.py — Self-Monitoring Module

KEY INSIGHT from results analysis:
The monitor was flagging TOO MANY answers (48%) and corrections
were making things worse. We need to be MORE SELECTIVE — only flag
answers we're highly confident are wrong.

New strategy:
  - Only flag clear uncertainty phrases (Check 1)
  - Only flag answers ungrounded via NLI model (Check 2, stricter)
  - Only flag empty/garbage answers (Check 3)
  - LLM verify ONLY when other checks pass AND answer seems suspicious
  - Skip LLM verify for short confident answers (saves API calls too)
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
import config
from config import LLM_BACKEND

logger = logging.getLogger(__name__)

UNCERTAINTY_PHRASES = [
    "i don't know", "i do not know", "cannot be determined",
    "not mentioned", "not provided", "no information",
    "i'm not sure", "not specified", "not available",
    "not enough information", "cannot answer", "cannot find",
    "the context does not", "there is no information",
    "based on the context, i cannot",
]


@dataclass
class FailureReport:
    """Report generated after evaluating an answer."""
    has_failure: bool = False
    confidence_ok: bool = True
    grounding_ok: bool = True
    completeness_ok: bool = True
    llm_verify_ok: bool = True
    consistency_ok: bool = True
    low_confidence: bool = False
    confidence_score: float = 1.0
    ungrounded_sentences: List[str] = field(default_factory=list)
    failure_reason: str = ""
    suggested_action: str = ""


class SelfMonitor:
    """Evaluates agent responses using multi-faceted checks and an NLI model."""

    def __init__(self, llm: Any) -> None:
        """
        Initialize the SelfMonitor.
        
        Args:
            llm: The LLM reasoner instance.
        """
        self.llm = llm
        self._nli_pipeline = None

    def _get_nli_pipeline(self) -> Any:
        """Lazily load the NLI pipeline."""
        if self._nli_pipeline is None:
            from transformers import pipeline
            model_name = getattr(config, 'NLI_MODEL', 'cross-encoder/nli-deberta-v3-small')
            logger.info("Loading NLI model for grounding checks: %s", model_name)
            self._nli_pipeline = pipeline("text-classification", model=model_name)
        return self._nli_pipeline

    def evaluate(self, question: str, answer: str, retrieved_docs: List[str], sub_answers: Optional[List[str]] = None) -> FailureReport:
        """
        Evaluate an answer against retrieved docs and original question.
        
        Args:
            question: The user query.
            answer: The generated answer.
            retrieved_docs: The context retrieved.
            sub_answers: Any sub-answers if decomposition was used.
            
        Returns:
            A FailureReport object containing evaluation results.
        """
        report = FailureReport()

        report.confidence_ok, report.low_confidence = self._check_confidence(answer)
        if report.low_confidence:
            report.confidence_score = 0.0

        report.completeness_ok = self._check_completeness(answer)
        if not report.completeness_ok:
            report.confidence_score = min(report.confidence_score, 0.2)

        if report.confidence_ok and report.completeness_ok:
            report.grounding_ok, report.ungrounded_sentences = self._check_grounding_nli(answer, retrieved_docs)
            if not report.grounding_ok:
                report.confidence_score = min(report.confidence_score, 0.4)

        if (report.confidence_ok and report.completeness_ok
                and report.grounding_ok
                and LLM_BACKEND in ("gemini", "groq", "openai", "ollama")
                and self._is_suspicious(answer, question)):
            report.llm_verify_ok = self._llm_verify(question, answer, retrieved_docs)
            if not report.llm_verify_ok:
                report.confidence_score = min(report.confidence_score, 0.5)

        if sub_answers and len(sub_answers) >= 2:
            report.consistency_ok = self._check_consistency(sub_answers)
            if not report.consistency_ok:
                report.confidence_score = min(report.confidence_score, 0.3)

        if not all([report.confidence_ok, report.grounding_ok,
                    report.completeness_ok, report.llm_verify_ok,
                    report.consistency_ok]):
            report.has_failure = True
            report.failure_reason = self._build_reason(report)
            report.suggested_action = self._suggest_action(report)

        return report

    def _check_confidence(self, answer: str) -> Tuple[bool, bool]:
        """Check for uncertainty phrases."""
        a = answer.lower().strip().rstrip(".")
        for phrase in UNCERTAINTY_PHRASES:
            if phrase in a:
                return False, True
        if not a:
            return False, True
        return True, False

    def _check_completeness(self, answer: str) -> bool:
        """Check if answer is empty or just noise."""
        a = answer.lower().strip()
        if not a or a in (".", ",", "?", "-", "n/a", "none", ""):
            return False
        for phrase in UNCERTAINTY_PHRASES:
            if a.startswith(phrase):
                return False
        return True

    def _check_grounding_nli(self, answer: str, retrieved_docs: List[str]) -> Tuple[bool, List[str]]:
        """
        NLI-based grounding check: ensures the answer is entailed by the retrieved docs.
        """
        if not retrieved_docs or len(answer.split()) <= 2:
            return True, []

        combined = " ".join(retrieved_docs)[:2000]
        nli = self._get_nli_pipeline()
        
        try:
            result = nli({"text": combined, "text_pair": answer})
            label = result.get('label', '').lower()
            if label == 'contradiction' or label == 'entailment_contradiction':
                return False, [answer]
        except Exception as e:
            logger.warning("NLI grounding check failed: %s", e)
            
        return True, []

    def _is_suspicious(self, answer: str, question: str) -> bool:
        """Only run LLM verify for answers that look suspicious."""
        a = answer.lower().strip()
        if a in ("yes", "no", "true", "false"):
            return False
        if re.match(r"^\d+$", a):
            return False
        if len(answer.split()) <= 2:
            return False
        return True

    def _llm_verify(self, question: str, answer: str, retrieved_docs: List[str]) -> bool:
        """LLM verification as a fallback check."""
        context = " ".join(d[:200] for d in retrieved_docs[:3])
        prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Context: {context}\n\n"
            f"Is this answer supported by the context? Reply YES or NO only."
        )
        try:
            response = self.llm.generate(prompt, max_tokens=5).strip().upper()
            return not response.startswith("NO")
        except Exception:
            return True

    def _check_consistency(self, sub_answers: List[str]) -> bool:
        """Check if sub-answers contradict each other."""
        lowers = [a.lower().strip().rstrip(".") for a in sub_answers]
        has_yes = any(a == "yes" or a.startswith("yes,") for a in lowers)
        has_no = any(a == "no" or a.startswith("no,") for a in lowers)
        return not (has_yes and has_no)

    def _build_reason(self, report: FailureReport) -> str:
        """Build a string describing why the check failed."""
        parts = []
        if not report.confidence_ok: parts.append("uncertain answer")
        if not report.grounding_ok: parts.append("answer not entailed by docs")
        if not report.completeness_ok: parts.append("empty/missing answer")
        if not report.llm_verify_ok: parts.append("LLM verification failed")
        if not report.consistency_ok: parts.append("sub-answers contradict")
        return "; ".join(parts)

    def _suggest_action(self, report: FailureReport) -> str:
        """Suggest a corrective action based on the failure report."""
        if not report.completeness_ok: return "decompose"
        if not report.grounding_ok: return "re_retrieve"
        if not report.llm_verify_ok: return "re_retrieve"
        if not report.confidence_ok: return "reformulate"
        return "reformulate"
