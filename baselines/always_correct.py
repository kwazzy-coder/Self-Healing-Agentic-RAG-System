"""
baselines/always_correct.py — Always-Correct baseline
Applies correction loop on EVERY query regardless of monitor output.
Proves that selective monitoring matters.
"""
import logging
import time
from typing import List, Dict, Any

from pipeline.pipeline import SelfHealingRAGPipeline

logger = logging.getLogger(__name__)

class AlwaysCorrectRAG(SelfHealingRAGPipeline):
    """
    Always applies correction loop regardless of monitor output.
    """
    def run(self, question: str, documents: List[str], **kwargs) -> Dict[str, Any]:
        """
        Run the always correct RAG pipeline.
        """
        start_time = time.time()
        self.retriever.index_documents(documents)
        
        retrieved_docs = self.retriever.retrieve(question)
        docs = [r.text for r in retrieved_docs]
        
        answer = self.llm.answer_with_context(question, docs)
        
        n_llm_calls = 1
        n_retrieval_calls = 1
        
        # Always trigger correction
        corrected_state = self.corrective_actions.apply({
            "question": question,
            "answer": answer,
            "context": docs,
            "failure_reason": "Always correct baseline triggered."
        })
        
        answer = corrected_state.get("answer", answer)
        n_llm_calls += corrected_state.get("n_llm_calls", 0)
        n_retrieval_calls += corrected_state.get("n_retrieval_calls", 0)
        
        end_time = time.time()
        
        return {
            "final_answer": answer.strip().rstrip("."),
            "retrieved_docs": corrected_state.get("context", docs),
            "failure_detected": True,
            "correction_attempted": True,
            "n_correction_rounds": self.corrective_actions.max_attempts,
            "failure_reason": "Always correct baseline triggered.",
            "trace": ["always_correct: correction applied"],
            "component_latencies": {"total": end_time - start_time},
            "n_llm_calls": n_llm_calls,
            "n_retrieval_calls": n_retrieval_calls,
        }
