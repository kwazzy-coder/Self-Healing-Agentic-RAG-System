"""
pipeline.py — Self-Healing Agentic RAG pipeline (orchestrator).

    plan -> retrieve -> generate -> [monitor -> correct]* -> select best candidate

Key behaviours
* Every candidate answer — including the one produced by the *last*
  correction — is checked by the monitor (the original loop never verified
  its final correction).
* The returned answer is the best-grounded candidate (passing > failing,
  then highest monitor confidence, then earliest), so a correction that makes
  things worse cannot replace a better earlier answer.
* The pipeline holds no per-query state (the document index, corrector and
  cost tracker are created per call), so one instance can serve concurrent
  API requests.
* Components are injected, which makes the system testable without network
  access and lets baselines share one copy of each model.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import config
from agent.corrective_actions import CorrectiveActions
from agent.cost_tracker import CURRENT_TRACKER, CostTracker
from agent.llm_reasoner import LLMReasoner
from agent.multihop import run_subquestions
from agent.nli import NLIScorer
from agent.self_monitor import FailureReport, MonitorSettings, SelfMonitor
from agent.task_planner import PlanResult, TaskPlanner
from retrieval.chunking import chunk_documents
from retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Behaviour switches. ``None`` means "use the value from config.py at run time"."""

    name: str = "self_healing"
    top_k: Optional[int] = None
    max_docs: Optional[int] = None
    max_correction_rounds: Optional[int] = None
    fusion_method: Optional[str] = None
    enable_planning: bool = True
    enable_healing: bool = True
    monitor_mode: str = "tiered"          # tiered | heuristic | nli | heuristic_nli | llm
    always_correct: bool = False          # baseline: correct once, no monitor
    short_answers: bool = True            # HotpotQA-style short answers vs. 1-3 sentence answers
    use_retrieval: bool = True            # False = closed-book LLM baseline
    chunk_size: int = 0                   # >0 splits long documents into chunks

    def resolved(self) -> PipelineConfig:
        return replace(
            self,
            top_k=self.top_k or config.TOP_K_RETRIEVAL,
            max_docs=self.max_docs or config.MAX_DOCS_PER_QUERY,
            max_correction_rounds=(config.MAX_CORRECTION_ATTEMPTS if self.max_correction_rounds is None
                                   else self.max_correction_rounds),
            fusion_method=self.fusion_method or config.RETRIEVAL_FUSION_METHOD,
        )


@dataclass
class Components:
    """Heavy, shareable objects (models/clients)."""

    llm: Any
    retriever: Any
    nli: Any

    @classmethod
    def build(cls, llm: Any = None, retriever: Any = None, nli: Any = None) -> Components:
        return cls(llm=llm or LLMReasoner(), retriever=retriever or HybridRetriever(), nli=nli or NLIScorer())


@dataclass
class RoundRecord:
    round: int
    answer: str
    produced_by: str            # "initial" or the corrective strategy
    has_failure: Optional[bool]
    failure_type: str = ""
    failure_reason: str = ""
    monitor_tier: int = 0
    confidence: Optional[float] = None
    n_docs: int = 0


@dataclass
class PipelineResult:
    """Everything the system did for one question."""

    question: str
    final_answer: str
    initial_answer: str
    retrieved_docs: List[str]
    strategy: str = "direct"
    sub_qas: List[Tuple[str, str]] = field(default_factory=list)
    failure_detected: bool = False
    correction_attempted: bool = False
    n_correction_rounds: int = 0
    failure_reason: str = ""
    failure_type: str = ""
    initial_monitor_tier: int = 0
    selected_round: int = 0
    verified: Optional[bool] = None   # True: final answer passed the monitor; None: no monitor ran
    rounds: List[RoundRecord] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
    component_latencies: Dict[str, float] = field(default_factory=dict)
    cost_data: Dict[str, Any] = field(default_factory=dict)
    system: str = "self_healing"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # Dict-style access for code written against the old dict return value.
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    @property
    def answer(self) -> str:  # alias used by older baseline code
        return self.final_answer


# Backwards-compatible alias: the baseline now returns the same structure.
BaselineResult = PipelineResult


@dataclass
class _Candidate:
    answer: str
    docs: List[str]
    report: Optional[FailureReport]
    round: int

    def sort_key(self) -> Tuple[int, float, int]:
        if self.report is None:
            return (1, 0.5, -self.round)
        return (int(not self.report.has_failure), self.report.confidence_score, -self.round)


class SelfHealingRAGPipeline:
    """Orchestrates planner, retriever, generator, monitor and corrector."""

    def __init__(
        self,
        pipeline_config: Optional[PipelineConfig] = None,
        llm: Any = None,
        retriever: Any = None,
        nli: Any = None,
        components: Optional[Components] = None,
    ) -> None:
        logger.info("Initialising Self-Healing RAG pipeline...")
        self.components = components or Components.build(llm=llm, retriever=retriever, nli=nli)
        self.config = pipeline_config or PipelineConfig()

    # Convenience accessors
    @property
    def llm(self) -> Any:
        return self.components.llm

    @property
    def retriever(self) -> Any:
        return self.components.retriever

    def _monitor(self, cfg: PipelineConfig) -> SelfMonitor:
        settings = MonitorSettings.from_mode(cfg.monitor_mode, short_answers=cfg.short_answers)
        if cfg.monitor_mode == "tiered":
            settings.use_nli = settings.use_nli and config.ENABLE_NLI
            settings.use_llm_judge = settings.use_llm_judge and config.ENABLE_LLM_JUDGE
        return SelfMonitor(self.llm, nli=self.components.nli, settings=settings)

    # ── Main entry point ─────────────────────────────────────────────────────
    def run(
        self,
        question: str,
        documents: List[str],
        question_type: Optional[str] = None,
        pipeline_config: Optional[PipelineConfig] = None,
    ) -> PipelineResult:
        cfg = (pipeline_config or self.config).resolved()
        if not cfg.use_retrieval:
            return self._closed_book(question, cfg)
        tracker = CostTracker()
        token = CURRENT_TRACKER.set(tracker)
        trace: List[str] = []
        try:
            result = self._run(question, documents, question_type, cfg, tracker, trace)
        finally:
            CURRENT_TRACKER.reset(token)
        cost = tracker.finalize_query()
        result.component_latencies = dict(cost.component_latencies)
        result.cost_data = cost.to_dict()
        return result

    def _run(self, question, documents, question_type, cfg: PipelineConfig, tracker: CostTracker,
             trace: List[str]) -> PipelineResult:
        # 1. Index
        with tracker.time_component("index"):
            docs = chunk_documents(documents, cfg.chunk_size) if cfg.chunk_size else list(documents)
            index = self.retriever.build_index(docs)
        trace.append(f"indexed {len(index)} passages")

        # 2. Plan
        with tracker.time_component("plan"):
            if cfg.enable_planning:
                planner = TaskPlanner(self.llm, enable_decomposition=config.ENABLE_DECOMPOSITION)
                plan = planner.plan(question, question_type)
            else:
                plan = PlanResult(strategy="direct", original=question)
        trace.append(f"plan: strategy={plan.strategy}, sub_questions={plan.sub_questions}")

        # 3. Retrieve
        with tracker.time_component("retrieve"):
            retrieved = index.search_texts(question, top_k=cfg.top_k, method=cfg.fusion_method)
            tracker.record_retrieval_call()
        trace.append(f"retrieved {len(retrieved)} passages")

        # 4. Generate
        sub_qas: List[Tuple[str, str]] = []
        with tracker.time_component("generate"):
            if plan.sub_questions:
                answer, context, sub_qas = run_subquestions(
                    self.llm, index, question, plan.sub_questions, plan.strategy, retrieved,
                    cfg.top_k, cfg.max_docs, short=cfg.short_answers,
                )
                for sq, sa in sub_qas:
                    trace.append(f"sub-answer: '{sq[:70]}' -> '{sa[:50]}'")
            elif plan.strategy != "direct":
                # Multi-hop without a usable decomposition: give the model a wider context.
                context = index.search_texts(question, top_k=cfg.max_docs, method=cfg.fusion_method)
                answer = self.llm.answer_with_context(question, context, short=cfg.short_answers)
            else:
                context = retrieved
                answer = self.llm.answer_with_context(question, context, short=cfg.short_answers)
        initial_answer = answer
        trace.append(f"initial answer: '{answer[:100]}'")

        result = PipelineResult(
            question=question, final_answer=answer, initial_answer=initial_answer, retrieved_docs=context,
            strategy=plan.strategy, sub_qas=sub_qas, system=cfg.name, trace=trace,
        )
        if not cfg.enable_healing and not cfg.always_correct:
            result.rounds = [RoundRecord(0, answer, "initial", None, n_docs=len(context))]
            result.final_answer = _tidy(answer)
            return result

        # 5. Self-healing loop
        corrector = CorrectiveActions(self.llm, top_k=cfg.top_k, max_docs=cfg.max_docs,
                                      short_answers=cfg.short_answers)
        if cfg.always_correct:
            return self._always_correct(result, corrector, index, question, question_type, tracker, trace)

        monitor = self._monitor(cfg)
        candidates: List[_Candidate] = []
        produced_by = "initial"
        cur_sub_qas, cur_strategy = list(sub_qas), plan.strategy
        for rnd in range(cfg.max_correction_rounds + 1):
            with tracker.time_component("monitor"):
                report = monitor.evaluate(question, answer, context,
                                          sub_answers=[a for _, a in cur_sub_qas] or None, strategy=cur_strategy)
            candidates.append(_Candidate(answer, context, report, rnd))
            result.rounds.append(RoundRecord(
                rnd, answer, produced_by, report.has_failure, report.failure_type, report.failure_reason,
                report.monitor_tier_used, round(report.confidence_score, 4), len(context),
            ))
            if rnd == 0:
                result.initial_monitor_tier = report.monitor_tier_used
            if not report.has_failure:
                trace.append(f"monitor round {rnd}: PASS (tier {report.monitor_tier_used}, "
                             f"confidence {report.confidence_score:.2f})")
                break
            result.failure_detected = True
            if not result.failure_reason:
                result.failure_reason, result.failure_type = report.failure_reason, report.failure_type
            trace.append(f"monitor round {rnd}: FAIL [{report.failure_type}] {report.failure_reason}")
            if rnd == cfg.max_correction_rounds:
                trace.append("correction budget exhausted")
                break

            with tracker.time_component("correction"):
                corr = corrector.apply(
                    action=report.suggested_action, question=question, previous_answer=answer,
                    retrieved_docs=context, ungrounded_sentences=report.ungrounded_sentences,
                    failure_type=report.failure_type, index=index, question_type=question_type,
                    sub_qas=cur_sub_qas,
                )
            if corr is None:
                trace.append("no corrective strategy left to try")
                break
            result.correction_attempted = True
            result.n_correction_rounds += 1
            answer, context, produced_by = corr.new_answer, corr.new_docs, corr.strategy
            # The next consistency check only applies if this answer was built from sub-questions.
            cur_sub_qas, cur_strategy = list(corr.sub_qas), corr.plan_strategy
            if corr.strategy == "decompose":
                result.sub_qas = cur_sub_qas
            trace.append(f"correction {result.n_correction_rounds}: {corr.action_taken} -> '{answer[:60]}'")

        best = max(candidates, key=_Candidate.sort_key)
        result.final_answer = _tidy(best.answer)
        result.retrieved_docs = best.docs
        result.selected_round = best.round
        result.verified = bool(best.report and not best.report.has_failure)
        if not result.verified:
            trace.append("no candidate passed the monitor: returning the best one, marked unverified")
        if len(candidates) > 1:
            trace.append(f"selected answer from round {best.round}: '{best.answer[:60]}'")
        return result

    def _always_correct(self, result, corrector, index, question, question_type, tracker, trace):
        """Baseline: apply one correction to every query without consulting the monitor."""
        result.rounds.append(RoundRecord(0, result.initial_answer, "initial", None,
                                         n_docs=len(result.retrieved_docs)))
        with tracker.time_component("correction"):
            corr = corrector.apply("re_retrieve", question, result.initial_answer, result.retrieved_docs,
                                   index=index, question_type=question_type, sub_qas=result.sub_qas)
        if corr is None:
            result.final_answer = _tidy(result.initial_answer)
            return result
        result.correction_attempted = True
        result.n_correction_rounds = 1
        result.rounds.append(RoundRecord(1, corr.new_answer, corr.strategy, None, n_docs=len(corr.new_docs)))
        result.final_answer = _tidy(corr.new_answer)
        result.retrieved_docs = corr.new_docs
        result.selected_round = 1
        trace.append(f"always-correct: {corr.action_taken} -> '{corr.new_answer[:60]}'")
        return result

    def _closed_book(self, question: str, cfg: PipelineConfig) -> PipelineResult:
        tracker = CostTracker()
        token = CURRENT_TRACKER.set(tracker)
        try:
            with tracker.time_component("generate"):
                answer = self.llm.answer_closed_book(question, short=cfg.short_answers)
        finally:
            CURRENT_TRACKER.reset(token)
        cost = tracker.finalize_query()
        return PipelineResult(
            question=question, final_answer=_tidy(answer), initial_answer=answer, retrieved_docs=[],
            rounds=[RoundRecord(0, answer, "closed_book", None)], trace=["closed-book answer (no retrieval)"],
            component_latencies=dict(cost.component_latencies), cost_data=cost.to_dict(), system=cfg.name,
        )

    def run_baseline(self, question: str, documents: List[str], question_type: Optional[str] = None) -> PipelineResult:
        """Vanilla RAG: retrieve + generate, no planning and no self-healing."""
        cfg = replace(self.config, name="vanilla_rag", enable_planning=False, enable_healing=False,
                      always_correct=False)
        return self.run(question, documents, question_type, pipeline_config=cfg)


def _tidy(answer: str) -> str:
    return (answer or "").strip().rstrip(".")
