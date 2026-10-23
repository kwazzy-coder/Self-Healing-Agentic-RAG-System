"""
agent/self_monitor.py — Three-tier cascading failure detector.

  Tier 1  Heuristics      ~0 ms, free   empty / refusal / answer-type mismatch / off-topic
  Tier 2  NLI grounding   ~50 ms, free  cross-encoder P(entailment) of the answer given the passages
  Tier 3  LLM judge       1 API call    only when Tier 2 is uncertain (NLI_LOW <= p <= NLI_HIGH)

Cheap tiers resolve the clear-cut cases, so the expensive LLM judge only runs
on the ambiguous ones. Each tier can be switched off for ablations
(``MonitorSettings``), which is how the "LLM-judge-only" and "NLI-only"
baselines are produced.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

import config
from agent.cost_tracker import CURRENT_TRACKER
from agent.nli import NLIScorer, answer_to_hypotheses, build_premises

logger = logging.getLogger(__name__)

UNCERTAINTY_PHRASES = [
    "i don't know", "i do not know", "cannot be determined", "can't be determined",
    "not mentioned", "not provided", "no information", "i'm not sure", "not specified",
    "not enough information", "cannot answer", "can't answer", "cannot find",
    "the context does not", "the passages do not", "there is no information",
    "based on the context, i cannot", "unable to determine",
]
_QUESTION_STOPWORDS = {
    "what", "who", "where", "when", "which", "how", "is", "are", "was", "were", "the",
    "a", "an", "does", "did", "do", "of", "in", "on", "for", "to", "and", "or", "by",
    "with", "that", "this", "from", "at", "as", "be", "has", "have", "had",
}
_AUX_START = re.compile(r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would|should)\b", re.I)

# Failure type -> recommended corrective strategy (consumed by CorrectiveActions)
FAILURE_STRATEGY_MAP = {
    "empty": "expand_retrieve",
    "refusal": "reformulate",
    "type_mismatch": "decompose",
    "irrelevant": "re_retrieve",
    "ungrounded": "re_retrieve",
    "hallucinated": "expand_retrieve",
    "inconsistent": "adopt_last_hop",
}


@dataclass
class MonitorSettings:
    use_heuristics: bool = True
    use_nli: bool = True
    use_llm_judge: bool = True
    short_answers: bool = True  # enables the yes/no answer-type check

    @classmethod
    def from_mode(cls, mode: str, short_answers: bool = True) -> MonitorSettings:
        modes = {
            "tiered": (True, True, True),
            "heuristic": (True, False, False),
            "nli": (False, True, False),
            "heuristic_nli": (True, True, False),
            "heuristic_llm": (True, False, True),
            "llm": (False, False, True),
        }
        if mode not in modes:
            raise ValueError(f"Unknown monitor mode '{mode}'. Choose from {sorted(modes)}")
        h, n, j = modes[mode]
        return cls(use_heuristics=h, use_nli=n, use_llm_judge=j, short_answers=short_answers)


@dataclass
class FailureReport:
    """Verdict of the monitor for one (question, answer, passages) triple."""

    has_failure: bool = False
    failure_type: str = ""
    failure_reason: str = ""
    suggested_action: str = ""
    monitor_tier_used: int = 0          # 0 = no check ran, 1/2/3 = deciding tier
    confidence_score: float = 0.5       # ordinal grounding confidence in [0, 1]
    nli_entailment: Optional[float] = None
    nli_contradiction: Optional[float] = None
    judge_verdict: Optional[bool] = None
    ungrounded_sentences: List[str] = field(default_factory=list)
    llm_calls: int = 0

    # Backwards-compatible flags used by older code / reports
    @property
    def grounding_ok(self) -> bool:
        return self.failure_type not in ("ungrounded", "hallucinated", "irrelevant")

    @property
    def confidence_ok(self) -> bool:
        return self.failure_type != "refusal"

    @property
    def completeness_ok(self) -> bool:
        return self.failure_type != "empty"

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).replace(" the ", " ").split())


def is_refusal(answer: str) -> Optional[str]:
    a = (answer or "").lower()
    return next((p for p in UNCERTAINTY_PHRASES if p in a), None)


def expects_yes_no(question: str) -> Optional[bool]:
    """True: yes/no question; False: 'A or B' choice question; None: open question."""
    q = question.strip().lower()
    if " or " in q:
        return False
    if _AUX_START.match(q):
        return True
    return None


class SelfMonitor:
    """Evaluates an answer with the cascading Tier 1 -> 2 -> 3 checks."""

    def __init__(
        self,
        llm: Any,
        nli: Optional[NLIScorer] = None,
        settings: Optional[MonitorSettings] = None,
        cost_tracker: Any = None,
    ) -> None:
        self.llm = llm
        self.nli = nli or NLIScorer()
        self.settings = settings or MonitorSettings(
            use_nli=config.ENABLE_NLI, use_llm_judge=config.ENABLE_LLM_JUDGE
        )
        self.cost_tracker = cost_tracker

    # ── Public API ───────────────────────────────────────────────────────────
    def evaluate(
        self,
        question: str,
        answer: str,
        retrieved_docs: List[str],
        sub_answers: Optional[List[str]] = None,
        strategy: Optional[str] = None,
    ) -> FailureReport:
        """
        Args:
            sub_answers: answers to the decomposed sub-questions, in order (if any).
            strategy: the plan used ("sequential" enables the multi-hop consistency check).
        """
        s = self.settings

        if s.use_heuristics:
            report = self._tier1(question, answer)
            if report is None and strategy == "sequential" and sub_answers:
                report = self._consistency(answer, sub_answers[-1])
            if report is not None:
                return self._finish(report)

        nli_score: Optional[float] = None
        report = FailureReport()
        if s.use_nli:
            nli_score, contradiction, weakest = self._tier2(question, answer, retrieved_docs)
            report.nli_entailment, report.nli_contradiction = nli_score, contradiction
            report.monitor_tier_used = 2
            report.confidence_score = nli_score
            if nli_score >= config.NLI_HIGH:
                return self._finish(report)
            if nli_score < config.NLI_LOW or (contradiction >= 0.9 and nli_score < 0.5):
                return self._finish(self._fail(
                    report, "ungrounded", f"NLI: answer not entailed by passages (p={nli_score:.2f})",
                    ungrounded=weakest,
                ))

        if s.use_llm_judge and self.llm is not None and getattr(self.llm, "is_api", False):
            verdict, reason = self._tier3(question, answer, retrieved_docs)
            report.monitor_tier_used = 3
            report.judge_verdict = verdict
            report.llm_calls += 1
            base = nli_score if nli_score is not None else 0.5
            if verdict is False:
                report.confidence_score = min(base, 0.5) / 2
                return self._finish(self._fail(report, "hallucinated", f"LLM judge: {reason}", ungrounded=[answer]))
            if verdict is True:
                report.confidence_score = max(base, 0.8)
                return self._finish(report)
            # Unparseable/failed verdict: fall through to the NLI-only decision below.

        # No (usable) judge: resolve the uncertain NLI band at its midpoint.
        if nli_score is not None and nli_score < (config.NLI_LOW + config.NLI_HIGH) / 2:
            return self._finish(self._fail(report, "ungrounded", f"NLI uncertain-low (p={nli_score:.2f})",
                                           ungrounded=[answer]))
        if report.monitor_tier_used == 0:
            report.monitor_tier_used = 1 if s.use_heuristics else 0
        return self._finish(report)

    # ── Tiers ────────────────────────────────────────────────────────────────
    def _tier1(self, question: str, answer: str) -> Optional[FailureReport]:
        report = FailureReport(monitor_tier_used=1, confidence_score=0.0)
        a = (answer or "").strip().lower().rstrip(".")

        if not a or a in {",", "?", "-", "n/a", "none", "null"}:
            return self._fail(report, "empty", "empty/missing answer")

        phrase = is_refusal(a)
        if phrase:
            return self._fail(report, "refusal", f"uncertain answer: '{phrase}'")

        if self.settings.short_answers:
            expected = expects_yes_no(question)
            is_yn = a in ("yes", "no")
            if expected is True and not is_yn and len(a.split()) <= 6:
                return self._fail(report, "type_mismatch", "yes/no question answered with an entity")
            if expected is False and is_yn:
                return self._fail(report, "type_mismatch", "choice question ('A or B') answered yes/no")

        q_words = {w.strip("?,.!\"'") for w in question.lower().split()} - _QUESTION_STOPWORDS
        a_words = {w.strip("?,.!\"'") for w in a.split()}
        if len(a.split()) > 12 and len(q_words) > 3 and not (q_words & a_words):
            report.confidence_score = 0.2
            return self._fail(report, "irrelevant", "long answer shares no keywords with the question")
        return None

    def _consistency(self, answer: str, last_hop: str) -> Optional[FailureReport]:
        """Bridge questions: the final answer should agree with the answer of the last hop."""
        if not last_hop or is_refusal(last_hop):
            return None
        a, b = _norm(answer), _norm(last_hop)
        if not a or not b or a == b or a in b or b in a:
            return None
        report = FailureReport(monitor_tier_used=1, confidence_score=0.1)
        return self._fail(report, "inconsistent", f"final answer '{answer}' contradicts last hop '{last_hop}'")

    def _tier2(self, question: str, answer: str, docs: List[str]):
        hypotheses = answer_to_hypotheses(question, answer)
        premises = build_premises(docs, config.NLI_MAX_PREMISE_CHARS)
        if not hypotheses or not premises:
            return 0.0, 0.0, [answer]
        try:
            scores = self.nli.score(premises, hypotheses)
        except Exception as err:  # model download / runtime failure -> neutral
            logger.warning("NLI check failed (%s); treating as uncertain.", err)
            return 0.5, 0.0, []
        tracker = self.cost_tracker or CURRENT_TRACKER.get()
        if tracker:
            tracker.record_nli_inference(len(premises) * len(hypotheses))
        entail = min(s.entailment for s in scores)          # weakest sentence decides
        contra = max(s.contradiction for s in scores)
        weakest = [s.hypothesis for s in scores if s.entailment < config.NLI_LOW] or \
                  [min(scores, key=lambda s: s.entailment).hypothesis]
        return entail, contra, weakest

    def _tier3(self, question: str, answer: str, docs: List[str]):
        try:
            return self.llm.judge_support(question, answer, docs)
        except Exception as err:
            logger.warning("LLM judge failed (%s); treating as uncertain.", err)
            return None, f"judge error: {err}"

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _fail(report: FailureReport, ftype: str, reason: str, ungrounded: Optional[List[str]] = None) -> FailureReport:
        report.has_failure = True
        report.failure_type = ftype
        report.failure_reason = reason
        report.suggested_action = FAILURE_STRATEGY_MAP.get(ftype, "re_retrieve")
        if ungrounded:
            report.ungrounded_sentences = list(ungrounded)
        return report

    def _finish(self, report: FailureReport) -> FailureReport:
        tracker = self.cost_tracker or CURRENT_TRACKER.get()
        if tracker:
            tracker.record_monitor_tier(report.monitor_tier_used)
        if report.has_failure:
            logger.info("Monitor tier %d flagged '%s': %s", report.monitor_tier_used,
                        report.failure_type, report.failure_reason)
        return report
