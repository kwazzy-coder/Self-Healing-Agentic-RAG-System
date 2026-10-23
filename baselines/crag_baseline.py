"""
baselines/crag_baseline.py — CRAG-style baseline (Yan et al., 2024, simplified).

The LLM grades each retrieved passage as CORRECT / INCORRECT / AMBIGUOUS.
Incorrect passages are dropped; if most are incorrect the question is
rephrased and retrieval is repeated over the remaining passages.
No decomposition and no multi-round correction.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import config
from agent import prompts
from agent.cost_tracker import CURRENT_TRACKER, CostTracker
from agent.multihop import merge_docs
from pipeline import Components, PipelineResult, RoundRecord, _tidy

logger = logging.getLogger(__name__)


class CRAGBaseline:
    def __init__(self, components: Optional[Components] = None, short_answers: bool = True) -> None:
        self.components = components or Components.build()
        self.short_answers = short_answers

    def _grade(self, question: str, doc: str) -> str:
        try:
            reply = self.components.llm.generate(prompts.doc_grade_prompt(question, doc), max_tokens=10).upper()
        except Exception as err:  # grading failure -> keep the passage
            logger.warning("CRAG grading failed: %s", err)
            return "AMBIGUOUS"
        if "INCORRECT" in reply:
            return "INCORRECT"
        if "CORRECT" in reply:
            return "CORRECT"
        return "AMBIGUOUS"

    def run(self, question: str, documents: List[str], question_type: Optional[str] = None) -> PipelineResult:
        tracker = CostTracker()
        token = CURRENT_TRACKER.set(tracker)
        llm = self.components.llm
        try:
            index = self.components.retriever.build_index(documents)
            docs = index.search_texts(question, top_k=config.TOP_K_RETRIEVAL)
            tracker.record_retrieval_call()
            initial = llm.answer_with_context(question, docs, short=self.short_answers)
            grades = [self._grade(question, d) for d in docs]
            kept = [d for d, g in zip(docs, grades, strict=False) if g != "INCORRECT"]
            n_bad = grades.count("INCORRECT")
            corrected = n_bad > len(docs) / 2
            if corrected:
                new_q = llm.rephrase(question) if llm.is_api else question
                extra = index.search_texts(new_q, top_k=config.TOP_K_RETRIEVAL, exclude=docs)
                tracker.record_retrieval_call()
                kept = merge_docs(kept, extra, config.MAX_DOCS_PER_QUERY)
            final_docs = kept or docs
            answer = llm.answer_with_context(question, final_docs, short=self.short_answers) \
                if (corrected or len(kept) != len(docs)) else initial
        finally:
            CURRENT_TRACKER.reset(token)
        cost = tracker.finalize_query()
        return PipelineResult(
            question=question, final_answer=_tidy(answer), initial_answer=initial, retrieved_docs=final_docs,
            failure_detected=n_bad > 0, correction_attempted=corrected, n_correction_rounds=int(corrected),
            failure_reason=f"{n_bad}/{len(docs)} passages graded INCORRECT" if n_bad else "",
            failure_type="irrelevant_docs" if n_bad else "",
            rounds=[RoundRecord(0, initial, "initial", None, n_docs=len(docs)),
                    *([RoundRecord(1, answer, "crag_regrade", None, n_docs=len(final_docs))]
                      if answer != initial else [])],
            selected_round=1 if answer != initial else 0,
            trace=[f"crag grades: {grades}"], component_latencies=dict(cost.component_latencies),
            cost_data=cost.to_dict(), system="crag",
        )
