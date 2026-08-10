"""
evaluation/faithfulness.py — RAG Triad Evaluation

Implements the three pillars of RAG evaluation:
1. Faithfulness: Is every claim in the answer supported by retrieved context?
2. Answer Relevancy: Does the answer address the question?
3. Context Relevancy: Is the retrieved context relevant to the question?

Uses LLM-as-judge for evaluation (works with free Gemini/Groq APIs).
"""

import json
from typing import List, Dict, Any

def compute_faithfulness(answer: str, context: List[str], llm: Any) -> float:
    """
    Computes faithfulness by extracting claims from the answer and verifying
    each claim against the context.
    """
    if not answer or not answer.strip():
        return 0.0
        
    # 1. Extract claims
    extract_prompt = f"""
    Given the following text, extract all the factual claims made.
    Output a JSON list of strings, where each string is a separate claim.
    
    Text: "{answer}"
    
    Output JSON list only:
    """
    
    try:
        claims_response = llm.generate(extract_prompt)
        # Attempt to parse json list from response
        # Using a simple heuristic to find the list structure if wrapped in markdown
        text = claims_response.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        claims = json.loads(text.strip())
        if not isinstance(claims, list):
            import re
            claims = [c.strip() for c in re.split(r'[.!?]+', answer) if c.strip()]
    except Exception:
        # Fallback if generation or parsing fails
        import re
        claims = [c.strip() for c in re.split(r'[.!?]+', answer) if c.strip()]
        
    if not claims:
        return 0.0
        
    # 2. Verify claims
    context_str = "\n\n".join(context)
    supported_count = 0
    
    for claim in claims:
        verify_prompt = f"""
        Given the following context and a claim, determine if the claim is fully supported by the context.
        Respond with ONLY 'YES' or 'NO'.
        
        Context:
        {context_str}
        
        Claim: {claim}
        """
        try:
            res = llm.generate(verify_prompt).strip().upper()
            if "YES" in res:
                supported_count += 1
        except Exception:
            pass
            
    return supported_count / len(claims)


def compute_answer_relevancy(answer: str, question: str, llm: Any) -> float:
    """
    Uses an LLM to judge if the generated answer actually answers the provided question.
    Returns 1.0 for yes, 0.0 for no.
    """
    if not answer or not answer.strip():
        return 0.0
        
    prompt = f"""
    Evaluate if the following answer directly addresses the provided question.
    It does not need to be perfectly accurate, but it should be attempting to answer the specific question.
    Respond with ONLY 'YES' or 'NO'.
    
    Question: {question}
    Answer: {answer}
    """
    try:
        res = llm.generate(prompt).strip().upper()
        if "YES" in res:
            return 1.0
        return 0.0
    except Exception:
        return 0.0


def compute_context_relevancy(context: List[str], question: str, llm: Any) -> float:
    """
    Evaluates what fraction of the retrieved passages are relevant to the question.
    """
    if not context:
        return 0.0
        
    relevant_count = 0
    for doc in context:
        prompt = f"""
        Evaluate if the following document contains any information relevant to answering the question.
        Respond with ONLY 'YES' or 'NO'.
        
        Question: {question}
        Document: {doc}
        """
        try:
            res = llm.generate(prompt).strip().upper()
            if "YES" in res:
                relevant_count += 1
        except Exception:
            pass
            
    return relevant_count / len(context)


def compute_rag_triad(answer: str, question: str, context: List[str], llm: Any) -> Dict[str, float]:
    """
    Computes all three RAG Triad metrics and returns a dictionary.
    """
    return {
        "faithfulness": compute_faithfulness(answer, context, llm),
        "answer_relevancy": compute_answer_relevancy(answer, question, llm),
        "context_relevancy": compute_context_relevancy(context, question, llm),
    }
