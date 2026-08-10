"""
baselines/llm_judge_only.py — LLM-Judge-Only baseline
Uses full LLM verification on EVERY query instead of tiered heuristics+NLI.
Proves that our tiered approach saves API calls while maintaining accuracy.
"""
import logging
from typing import List, Dict, Any
import time

from pipeline.pipeline import SelfHealingRAGPipeline

logger = logging.getLogger(__name__)

class LLMJudgeOnlyRAG(SelfHealingRAGPipeline):
    """
    Uses LLM-as-judge for EVERY query (no tiered monitoring).
    """
    def run(self, question: str, documents: List[str], **kwargs) -> Dict[str, Any]:
        """
        Run the LLM Judge Only RAG pipeline.
        """
        start_time = time.time()
        self.retriever.index_documents(documents)
        
        retrieved_docs = self.retriever.retrieve(question)
        docs = [r.text for r in retrieved_docs]
        
        answer = self.llm.answer_with_context(question, docs)
        
        n_llm_calls = 1
        n_retrieval_calls = 1
        
        # Always use LLM verification (bypass Tier 1 and 2)
        verification = self.self_monitor._llm_verify(question, answer, docs)
        n_llm_calls += 1
        
        failure_detected = not verification["is_correct"]
        failure_reason = verification.get("reason", "")
        
        correction_attempted = False
        n_correction_rounds = 0
        
        if failure_detected:
            correction_attempted = True
            corrected_state = self.corrective_actions.apply({
                "question": question,
                "answer": answer,
                "context": docs,
                "failure_reason": failure_reason
            })
            answer = corrected_state.get("answer", answer)
            n_llm_calls += corrected_state.get("n_llm_calls", 0)
            n_retrieval_calls += corrected_state.get("n_retrieval_calls", 0)
            n_correction_rounds = 1
            
        end_time = time.time()
        
        return {
            "final_answer": answer.strip().rstrip("."),
            "retrieved_docs": docs,
            "failure_detected": failure_detected,
            "correction_attempted": correction_attempted,
            "n_correction_rounds": n_correction_rounds,
            "failure_reason": failure_reason,
            "trace": ["llm_judge_only: completed"],
            "component_latencies": {"total": end_time - start_time},
            "n_llm_calls": n_llm_calls,
            "n_retrieval_calls": n_retrieval_calls,
        }
