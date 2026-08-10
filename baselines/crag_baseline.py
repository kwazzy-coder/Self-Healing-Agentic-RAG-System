"""
baselines/crag_baseline.py — CRAG-style baseline
Reimplements the core CRAG idea: use LLM to grade retrieved documents
as Correct/Incorrect/Ambiguous, then re-retrieve if score is low.
No decomposition or multi-round correction.
"""
import logging
from typing import List, Dict, Any
import time

from agent.llm_reasoner import LLMReasoner
from retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)

class CRAGBaseline:
    """
    CRAG-style baseline for evaluation.
    """
    def __init__(self):
        self.llm = LLMReasoner()
        self.retriever = HybridRetriever()
        
    def _grade_doc(self, question: str, doc: str) -> str:
        prompt = f"Is the following document relevant to answering the question? Reply ONLY with CORRECT, INCORRECT, or AMBIGUOUS.\n\nQuestion: {question}\n\nDocument: {doc}"
        try:
            # Assuming the LLM reasoner has a generate method or we use answer_with_context 
            if hasattr(self.llm, "generate"):
                response = self.llm.generate(prompt).strip().upper()
            else:
                response = self.llm.answer_with_context(prompt, []).strip().upper()
        except Exception:
            response = "AMBIGUOUS"
            
        if "CORRECT" in response and "INCORRECT" not in response:
            return "CORRECT"
        elif "INCORRECT" in response:
            return "INCORRECT"
        else:
            return "AMBIGUOUS"

    def run(self, question: str, documents: List[str], **kwargs) -> Dict[str, Any]:
        """
        Run the CRAG pipeline.
        """
        start_time = time.time()
        self.retriever.index_documents(documents)
        
        retrieved_docs = self.retriever.retrieve(question)
        docs = [r.text for r in retrieved_docs]
        n_retrieval_calls = 1
        n_llm_calls = 0
        
        grades = []
        for doc in docs:
            grades.append(self._grade_doc(question, doc))
            n_llm_calls += 1
            
        correct_count = grades.count("CORRECT")
        incorrect_count = grades.count("INCORRECT")
        
        final_docs = [doc for doc, grade in zip(docs, grades) if grade != "INCORRECT"]
        
        if incorrect_count > len(docs) / 2:
            # Majority incorrect, re-retrieve
            modified_query = f"{question} (clarification needed)"
            new_retrieved = self.retriever.retrieve(modified_query)
            n_retrieval_calls += 1
            final_docs.extend([r.text for r in new_retrieved])
            
        if not final_docs:
            final_docs = docs # Fallback
            
        answer = self.llm.answer_with_context(question, final_docs)
        n_llm_calls += 1
        
        end_time = time.time()
        
        return {
            "final_answer": answer.strip().rstrip("."),
            "retrieved_docs": final_docs,
            "failure_detected": incorrect_count > 0,
            "correction_attempted": incorrect_count > len(docs) / 2,
            "n_correction_rounds": 1 if incorrect_count > len(docs) / 2 else 0,
            "failure_reason": "CRAG doc grading",
            "trace": [f"crag: {correct_count} correct, {incorrect_count} incorrect"],
            "component_latencies": {"total": end_time - start_time},
            "n_llm_calls": n_llm_calls,
            "n_retrieval_calls": n_retrieval_calls,
        }
