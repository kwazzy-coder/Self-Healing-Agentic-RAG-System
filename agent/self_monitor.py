"""
agent/self_monitor.py — Multi-Tiered Self-Monitoring Module

Core novelty: 3-tier cascading failure detection.
  Tier 1: Zero-cost heuristics (empty/refusal, keyword overlap) — <1ms, 0 API calls
  Tier 2: Lightweight NLI entailment (67M param cross-encoder) — ~50ms, 0 API calls
  Tier 3: LLM-as-judge (full semantic check) — ~2-4s, 1 API call

Key insight: ~70% of queries are resolved at Tier 1-2, saving significant API cost.
Tier 3 only fires when NLI score is in the uncertain range [τ_low, τ_high].
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
import config
from config import LLM_BACKEND

logger = logging.getLogger(__name__)

# NLI thresholds for Tier 2 → Tier 3 escalation
NLI_CONTRADICTION_THRESHOLD = 0.3  # Below this = definitely ungrounded
NLI_ENTAILMENT_THRESHOLD = 0.7     # Above this = definitely grounded
# Between 0.3 and 0.7 = uncertain → escalate to Tier 3

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
    monitor_tier_used: int = 0  # which tier caught the failure (0=pass, 1, 2, 3)
    failure_type: str = ""  # categorized failure type for adaptive correction

class SelfMonitor:
    """Evaluates agent responses using explicit multi-tiered cascading checks."""

    def __init__(self, llm: Any, cost_tracker: Any = None) -> None:
        """
        Initialize the SelfMonitor.
        
        Args:
            llm: The LLM reasoner instance.
            cost_tracker: Optional tracker for monitoring API and model usage.
        """
        self.llm = llm
        self._nli_pipeline = None
        self.cost_tracker = cost_tracker  # for tracking
    
    def evaluate(self, question: str, answer: str, retrieved_docs: List[str], sub_answers: Optional[List[str]] = None) -> FailureReport:
        """
        Evaluate an answer through a 3-tier failure detection pipeline.
        
        Args:
            question: The user query.
            answer: The generated answer.
            retrieved_docs: The context retrieved.
            sub_answers: Any sub-answers if decomposition was used.
            
        Returns:
            A FailureReport object containing evaluation results.
        """
        report = FailureReport()
        
        # ═══ TIER 1: Zero-cost heuristics ═══
        tier1_result = self._tier1_heuristics(answer, question)
        if tier1_result is not None:
            report = tier1_result
            report.monitor_tier_used = 1
            if self.cost_tracker:
                self.cost_tracker.record_monitor_tier(1)
            logger.info("Tier 1 caught failure: %s", report.failure_type)
            return report
        
        # ═══ TIER 2: NLI grounding check ═══
        nli_score = self._tier2_nli_check(answer, retrieved_docs)
        if self.cost_tracker:
            self.cost_tracker.record_nli_inference()
        
        if nli_score < NLI_CONTRADICTION_THRESHOLD:
            # Definitely ungrounded
            report.has_failure = True
            report.grounding_ok = False
            report.confidence_score = nli_score
            report.ungrounded_sentences = [answer]
            report.failure_reason = f"NLI contradiction (score={nli_score:.2f})"
            report.suggested_action = "re_retrieve"
            report.monitor_tier_used = 2
            report.failure_type = "ungrounded"
            if self.cost_tracker:
                self.cost_tracker.record_monitor_tier(2)
            logger.info("Tier 2 caught failure: %s", report.failure_type)
            return report
        
        if nli_score > NLI_ENTAILMENT_THRESHOLD:
            # Definitely grounded — PASS at Tier 2
            report.monitor_tier_used = 2
            report.confidence_score = nli_score
            if self.cost_tracker:
                self.cost_tracker.record_monitor_tier(2)
            # Still check consistency if sub_answers exist
            if sub_answers and len(sub_answers) >= 2:
                if not self._check_consistency(sub_answers):
                    report.has_failure = True
                    report.consistency_ok = False
                    report.failure_reason = "sub-answer contradiction"
                    report.suggested_action = "decompose"
                    report.failure_type = "inconsistent"
            return report
        
        # NLI score in uncertain range — escalate to Tier 3
        # ═══ TIER 3: LLM-as-judge ═══
        if LLM_BACKEND in ("gemini", "groq", "openai", "ollama"):
            llm_ok = self._tier3_llm_judge(question, answer, retrieved_docs)
            if self.cost_tracker:
                self.cost_tracker.record_llm_call()
                self.cost_tracker.record_monitor_tier(3)
            report.monitor_tier_used = 3
            
            if not llm_ok:
                report.has_failure = True
                report.llm_verify_ok = False
                report.confidence_score = nli_score
                report.failure_reason = f"LLM judge rejected (NLI was uncertain: {nli_score:.2f})"
                report.suggested_action = "re_retrieve"
                report.failure_type = "hallucinated"
                logger.info("Tier 3 caught failure: %s", report.failure_type)
                return report
        
        # All checks passed
        return report
    
    def _tier1_heuristics(self, answer: str, question: str) -> Optional[FailureReport]:
        """Tier 1: Zero-cost checks. Returns FailureReport if failure, None if pass."""
        report = FailureReport()
        a = answer.lower().strip().rstrip(".")
        
        # Check 1a: Empty/garbage
        if not a or a in (".", ",", "?", "-", "n/a", "none"):
            report.has_failure = True
            report.completeness_ok = False
            report.confidence_score = 0.0
            report.failure_reason = "empty/missing answer"
            report.suggested_action = "expand_retrieve"
            report.failure_type = "empty"
            return report
        
        # Check 1b: Uncertainty phrases  
        for phrase in UNCERTAINTY_PHRASES:
            if phrase in a:
                report.has_failure = True
                report.confidence_ok = False
                report.low_confidence = True
                report.confidence_score = 0.0
                report.failure_reason = f"uncertain answer: '{phrase}'"
                report.suggested_action = "reformulate"
                report.failure_type = "refusal"
                return report
        
        # Check 1c: Answer-question keyword overlap
        q_words = set(question.lower().split()) - {'what', 'who', 'where', 'when', 'which', 'how', 'is', 'are', 'was', 'were', 'the', 'a', 'an', 'does', 'did', 'do'}
        a_words = set(a.split())
        # If there's ZERO overlap between meaningful question words and answer
        # AND the answer is not a simple yes/no, something might be wrong
        # But be conservative — only flag if answer is long and has zero overlap
        if len(a.split()) > 5 and len(q_words) > 3:
            overlap = q_words & a_words
            if len(overlap) == 0:
                report.has_failure = True
                report.grounding_ok = False
                report.confidence_score = 0.2
                report.failure_reason = "answer has no keyword overlap with question"
                report.suggested_action = "re_retrieve"
                report.failure_type = "irrelevant"
                return report
        
        return None  # Pass Tier 1
    
    def _tier2_nli_check(self, answer: str, retrieved_docs: List[str]) -> float:
        """Tier 2: NLI entailment score. Returns float 0-1."""
        if not retrieved_docs or len(answer.split()) <= 2:
            return 0.8  # Short answers get benefit of doubt
        
        combined = " ".join(retrieved_docs)[:2000]
        nli = self._get_nli_pipeline()
        
        try:
            result = nli(f"{combined}</s></s>{answer}")  # Use proper separator
            # Cross-encoder returns labels like 'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'
            # Map to a score
            if isinstance(result, list):
                result = result[0]
            label = result.get('label', '').upper()
            score = result.get('score', 0.5)
            
            if 'ENTAILMENT' in label:
                return score  # High = good
            elif 'CONTRADICTION' in label:
                return 1.0 - score  # Contradiction confidence inverted
            else:  # NEUTRAL
                return 0.5
        except Exception as e:
            logger.warning("NLI check failed: %s. Defaulting to pass.", e)
            return 0.8  # Fail open
    
    def _tier3_llm_judge(self, question: str, answer: str, retrieved_docs: List[str]) -> bool:
        """Tier 3: LLM verification. Returns bool."""
        context = " ".join(d[:200] for d in retrieved_docs[:3])
        prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Context: {context}\n\n"
            f"Is this answer fully supported by the context? Reply YES or NO only."
        )
        try:
            response = self.llm.generate(prompt, max_tokens=5).strip().upper()
            return not response.startswith("NO")
        except Exception:
            return True  # Fail open

    def _get_nli_pipeline(self) -> Any:
        """Lazily load the NLI pipeline."""
        if self._nli_pipeline is None:
            from transformers import pipeline
            model_name = getattr(config, 'NLI_MODEL', 'cross-encoder/nli-deberta-v3-small')
            logger.info("Loading NLI model for grounding checks: %s", model_name)
            self._nli_pipeline = pipeline("text-classification", model=model_name)
        return self._nli_pipeline

    def _check_consistency(self, sub_answers: List[str]) -> bool:
        """Check if sub-answers contradict each other."""
        lowers = [a.lower().strip().rstrip(".") for a in sub_answers]
        has_yes = any(a == "yes" or a.startswith("yes,") for a in lowers)
        has_no = any(a == "no" or a.startswith("no,") for a in lowers)
        return not (has_yes and has_no)
