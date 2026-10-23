"""
agent/prompts.py — All prompt templates in one place.

Keeping prompts centralised makes them easy to review, version and ablate,
and keeps the agent logic free of long string literals.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

COT_FEW_SHOT = """Examples of correct reasoning:

Q: What nationality is the director of the film Titanic?
Context: Titanic is a 1997 film directed by James Cameron. James Cameron was born in Kapuskasing, Ontario, Canada.
Reasoning: The film Titanic was directed by James Cameron. James Cameron was born in Canada, so he is Canadian.
Answer: Canadian

Q: Were Scott Derrickson and Ed Wood both film directors?
Context: Scott Derrickson is an American director known for Doctor Strange. Ed Wood was an American filmmaker known for Plan 9 from Outer Space.
Reasoning: Scott Derrickson is described as a director. Ed Wood is described as a filmmaker. Both are directors.
Answer: yes

Q: Which magazine was started first, Time or Newsweek?
Context: Time magazine was first published on March 3, 1923. Newsweek was first published on February 17, 1933.
Reasoning: Time was published in 1923. Newsweek was published in 1933. 1923 comes before 1933.
Answer: Time

"""


def format_passages(passages: Sequence[str], char_limit: int) -> str:
    """
    Number the passages and share the character budget fairly between them.

    The original implementation joined everything and cut at a fixed length,
    which silently dropped the last retrieved documents — often the second
    hop of a multi-hop question.
    """
    passages = [p.strip() for p in passages if p and p.strip()]
    if not passages:
        return "(no passages)"
    per_doc = max(300, char_limit // len(passages))
    lines = []
    for i, p in enumerate(passages, 1):
        text = p if len(p) <= per_doc else p[: per_doc - 3].rstrip() + "..."
        lines.append(f"[{i}] {text}")
    return "\n\n".join(lines)


def short_answer_prompt(question: str, context: str) -> str:
    return (
        f"{COT_FEW_SHOT}"
        "Now answer this question using ONLY the context below.\n"
        "First write your Reasoning (1-2 sentences), then write Answer "
        "(as short as possible: a name, date, number, short phrase, or yes/no).\n"
        "If the context does not contain the answer, write: Answer: I don't know\n\n"
        f"Context:\n{context}\n\n"
        f"Q: {question}\n"
        "Reasoning:"
    )


def closed_book_prompt(question: str, short: bool = True) -> str:
    if short:
        return (
            f"{COT_FEW_SHOT}"
            "Answer from your own knowledge (no context is available).\n"
            "First write Reasoning (1-2 sentences), then Answer (as short as possible).\n\n"
            f"Q: {question}\n"
            "Reasoning:"
        )
    return f"Answer the question in 1-3 concise sentences.\n\nQuestion: {question}\nAnswer:"


def long_answer_prompt(question: str, context: str) -> str:
    return (
        "Answer the question using ONLY the context passages below. "
        "Write 1-3 concise sentences. Do not add facts that are not in the context. "
        "If the context does not contain the answer, say \"I don't know\".\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def synthesis_prompt(
    question: str, sub_qas: List[Tuple[str, str]], context: str, short: bool = True
) -> str:
    findings = "\n".join(f"- {q} -> {a}" for q, a in sub_qas) or "- (none)"
    if not short:
        return (
            "Answer the question using the intermediate findings and the context. "
            "Write 1-3 concise sentences grounded in the context.\n\n"
            f"Findings:\n{findings}\n\nContext:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
    return (
        f"{COT_FEW_SHOT}"
        "Use the intermediate findings and the context to answer.\n"
        "First write Reasoning (1-2 sentences), then Answer (as short as possible).\n\n"
        f"Findings:\n{findings}\n\n"
        f"Context:\n{context}\n\n"
        f"Q: {question}\n"
        "Reasoning:"
    )


def decompose_prompt(question: str, strategy: str) -> str:
    if strategy == "parallel":
        return (
            "Break this comparison question into 2 independent sub-questions, "
            "one for each entity being compared.\n\n"
            "Example:\n"
            "Question: Which magazine was started first, Time or Newsweek?\n"
            "1. When was Time magazine first published?\n"
            "2. When was Newsweek first published?\n\n"
            f"Question: {question}\n"
            "Output exactly 2 lines numbered \"1.\" and \"2.\" and nothing else."
        )
    return (
        "Break this multi-hop question into 2 simpler sub-questions that must be "
        "answered in order. In sub-question 2, write #1 to refer to the answer of "
        "sub-question 1.\n\n"
        "Example:\n"
        "Question: What nationality is the director of the film Titanic?\n"
        "1. Who directed the film Titanic?\n"
        "2. What nationality is #1?\n\n"
        f"Question: {question}\n"
        "Output exactly 2 lines numbered \"1.\" and \"2.\" and nothing else."
    )


def judge_prompt(question: str, answer: str, context: str) -> str:
    return (
        "You are a strict fact-checker. Decide whether the ANSWER to the QUESTION is "
        "fully supported by the CONTEXT passages. Use only the context, not outside "
        "knowledge. A short answer is supported if the context lets you derive it.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n"
        f"ANSWER: {answer}\n\n"
        "Reply with exactly one line:\n"
        "VERDICT: SUPPORTED - <short reason>\n"
        "or\n"
        "VERDICT: UNSUPPORTED - <short reason>"
    )


def search_query_prompt(question: str, previous_answer: str, unsupported: Sequence[str]) -> str:
    gaps = "; ".join(s for s in unsupported if s) or "unknown"
    return (
        "A retrieval system failed to find enough evidence to answer a question.\n"
        f"Question: {question}\n"
        f"Current answer (may be wrong): {previous_answer or '(none)'}\n"
        f"Unsupported part: {gaps}\n\n"
        "Write ONE short search query (at most 12 words) that would find the missing "
        "information. Output only the query."
    )


def rephrase_prompt(question: str) -> str:
    return (
        "Rephrase this question using different keywords but the same meaning.\n"
        f"Question: {question}\n"
        "Output only the rephrased question."
    )


def doc_grade_prompt(question: str, doc: str) -> str:
    return (
        "Is the following document relevant to answering the question? "
        "Reply ONLY with CORRECT, INCORRECT, or AMBIGUOUS.\n\n"
        f"Question: {question}\n\nDocument: {doc}"
    )
