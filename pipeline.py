"""
pipeline.py — Self-Healing Agentic RAG Pipeline

KEY FIX: Track best answer across all rounds.
If corrections make things worse, return the best answer seen.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from agent.llm_reasoner import LLMReasoner
from agent.task_planner import TaskPlanner
from agent.self_monitor import SelfMonitor
from agent.corrective_actions import CorrectiveActions
from agent.cost_tracker import CostTracker
from retrieval.retriever import HybridRetriever
from config import MAX_CORRECTION_ATTEMPTS

try:
    from config import MAX_DOCS_PER_QUERY
except ImportError:
    MAX_DOCS_PER_QUERY = 8

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Dataclass holding the result of a self-healing RAG execution."""
    final_answer: str
    retrieved_docs: List[str]
    failure_detected: bool
    correction_attempted: bool
    n_correction_rounds: int
    failure_reason: str
    trace: List[str]
    component_latencies: Dict[str, float] = field(default_factory=dict)
    cost_data: dict = field(default_factory=dict)


@dataclass
class BaselineResult:
    """Dataclass holding the result of a baseline RAG execution."""
    answer: str
    retrieved_docs: List[str]
    trace: List[str]
    component_latencies: Dict[str, float] = field(default_factory=dict)


class SelfHealingRAGPipeline:
    """Self-healing RAG Pipeline orchestration."""

    def __init__(self) -> None:
        """Initializes pipeline components."""
        logger.info("[Pipeline] Initialising Self-Healing RAG...")
        self.llm = LLMReasoner()
        self.retriever = HybridRetriever()
        self.cost_tracker = CostTracker()
        self.planner = TaskPlanner(self.llm)
        self.monitor = SelfMonitor(self.llm, cost_tracker=self.cost_tracker)
        self.corrector = CorrectiveActions(self.llm, self.retriever, cost_tracker=self.cost_tracker)
        logger.info("[Pipeline] Ready.")

    def run(self, question: str, documents: List[str], question_type: Optional[str] = None) -> PipelineResult:
        """
        Runs the self-healing RAG pipeline on a question and document set.
        
        Args:
            question: The user query.
            documents: A list of background documents to index.
            question_type: The type of question (optional).
            
        Returns:
            A PipelineResult object.
        """
        trace = []
        latencies = {}
        self.corrector.reset_round()
        self.cost_tracker.reset_query()

        # Step 1: Index
        with self.cost_tracker.time_component("index"):
            self.retriever.index_documents(documents)

        # Step 2: Plan
        with self.cost_tracker.time_component("plan"):
            plan = self.planner.plan(question, question_type)
        trace.append(f"plan: strategy={plan['strategy']}, sub_qs={plan['sub_questions']}")

        # Step 3: Retrieve
        with self.cost_tracker.time_component("retrieve"):
            retrieved = self.retriever.retrieve(question)
            self.cost_tracker.record_retrieval_call()
        # Handle RetrievalResult objects instead of dicts now
        retrieved_docs = [r.text if hasattr(r, 'text') else r["text"] for r in retrieved]
        trace.append(f"retrieved {len(retrieved_docs)} docs")

        # Step 4: Generate
        with self.cost_tracker.time_component("generate"):
            sub_answers = []

            if plan["strategy"] != "direct" and plan["sub_questions"]:
                for sq in plan["sub_questions"]:
                    extra = self.retriever.retrieve(sq)
                    self.cost_tracker.record_retrieval_call()
                    extra_docs = [r.text if hasattr(r, 'text') else r["text"] for r in extra]
                    all_docs = list(dict.fromkeys(retrieved_docs + extra_docs))[:MAX_DOCS_PER_QUERY]
                    sa = self.llm.answer_with_context(sq, all_docs)
                    self.cost_tracker.record_llm_call()
                    sub_answers.append(sa)
                    trace.append(f"sub-answer: '{sq[:50]}' -> '{sa[:50]}'")
                answer = self.llm.answer_with_sub_answers(question, sub_answers, retrieved_docs)
                self.cost_tracker.record_llm_call()

            elif plan["strategy"] != "direct" and not plan["sub_questions"]:
                all_retrieved = self.retriever.retrieve(question, top_k=len(documents))
                self.cost_tracker.record_retrieval_call()
                retrieved_docs = [r.text if hasattr(r, 'text') else r["text"] for r in all_retrieved]
                answer = self.llm.answer_with_context(question, retrieved_docs)
                self.cost_tracker.record_llm_call()
                trace.append(f"multi-hop direct: using {len(retrieved_docs)} docs")
            else:
                answer = self.llm.answer_with_context(question, retrieved_docs)
                self.cost_tracker.record_llm_call()

        trace.append(f"initial answer: '{answer[:100]}'")

        # KEY: track best answer seen across all rounds
        best_answer = answer

        # Step 5: Self-monitoring + correction
        failure_detected = False
        correction_attempted = False
        failure_reason = ""
        n_rounds = 0

        monitor_latency = 0.0
        correction_latency = 0.0

        for attempt in range(MAX_CORRECTION_ATTEMPTS):
            with self.cost_tracker.time_component("monitor"):
                report = self.monitor.evaluate(
                    question=question,
                    answer=answer,
                    retrieved_docs=retrieved_docs,
                    sub_answers=sub_answers or None,
                )

            if not report.has_failure:
                trace.append(f"self-monitor: PASS (attempt {attempt})")
                best_answer = answer
                break

            failure_detected = True
            failure_reason = report.failure_reason
            trace.append(
                f"self-monitor: FAIL - {report.failure_reason} "
                f"-> action={report.suggested_action}"
            )

            correction_attempted = True
            with self.cost_tracker.time_component("correction"):
                correction = self.corrector.apply(
                    action=report.suggested_action,
                    question=question,
                    previous_answer=answer,
                    retrieved_docs=retrieved_docs,
                    ungrounded_sentences=report.ungrounded_sentences,
                )

            new_answer = correction["new_answer"]
            retrieved_docs = correction["new_docs"]
            n_rounds += 1
            trace.append(f"correction {n_rounds}: {correction['action_taken']} -> '{new_answer[:60]}'")

            # Only update answer if correction helped
            if new_answer and new_answer.lower() != "i don't know":
                answer = new_answer
                best_answer = new_answer  # track best
                
        cost = self.cost_tracker.finalize_query()

        return PipelineResult(
            final_answer=best_answer.strip().rstrip("."),
            retrieved_docs=retrieved_docs,
            failure_detected=failure_detected,
            correction_attempted=correction_attempted,
            n_correction_rounds=n_rounds,
            failure_reason=failure_reason,
            trace=trace,
            component_latencies=cost.component_latencies,
            cost_data=cost.to_dict()
        )

    def run_baseline(self, question: str, documents: List[str]) -> BaselineResult:
        """
        Runs a simple baseline RAG without self-healing.
        
        Args:
            question: The user query.
            documents: A list of background documents to index.
            
        Returns:
            A BaselineResult object.
        """
        trace = []
        latencies = {}

        start_t = time.time()
        self.retriever.index_documents(documents)
        latencies["index"] = time.time() - start_t
        trace.append(f"indexed {len(documents)} docs")

        start_t = time.time()
        retrieved = self.retriever.retrieve(question)
        docs = [r.text if hasattr(r, 'text') else r["text"] for r in retrieved]
        latencies["retrieve"] = time.time() - start_t
        trace.append(f"retrieved {len(docs)} docs")

        start_t = time.time()
        answer = self.llm.answer_with_context(question, docs)
        latencies["generate"] = time.time() - start_t
        trace.append(f"baseline answer generated")

        return BaselineResult(
            answer=answer.strip().rstrip("."),
            retrieved_docs=docs,
            trace=trace,
            component_latencies=latencies
        )
