"""
baselines/vanilla_rag.py — Standard RAG baseline (no self-healing)
"""
import logging
from typing import List, Dict, Any
from agent.llm_reasoner import LLMReasoner
from retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)

class VanillaRAG:
    """
    Standard retrieve-and-generate with NO correction loop.
    """
    def __init__(self):
        self.llm = LLMReasoner()
        self.retriever = HybridRetriever()
    
    def run(self, question: str, documents: List[str], **kwargs) -> Dict[str, Any]:
        """
        Run the vanilla RAG pipeline.
        
        Args:
            question: The user query.
            documents: List of documents to index.
            **kwargs: Additional arguments.
            
        Returns:
            Dict containing the final answer and metadata.
        """
        self.retriever.index_documents(documents)
        retrieved = self.retriever.retrieve(question)
        docs = [r.text for r in retrieved]
        answer = self.llm.answer_with_context(question, docs)
        return {
            "final_answer": answer.strip().rstrip("."),
            "retrieved_docs": docs,
            "failure_detected": False,
            "correction_attempted": False,
            "n_correction_rounds": 0,
            "failure_reason": "",
            "trace": [f"vanilla_rag: retrieved {len(docs)} docs"],
            "component_latencies": {},
            "n_llm_calls": 1,
            "n_retrieval_calls": 1,
        }
