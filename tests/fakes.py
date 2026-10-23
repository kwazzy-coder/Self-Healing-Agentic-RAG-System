"""
tests/fakes.py — Offline stand-ins for the network/model-backed components.

* HashEncoder     — bag-of-words hashing "embedding model" (no download)
* FakeNLIModel    — lexical-overlap NLI cross-encoder with the real label layout
* FakeChatClient  — OpenAI-compatible client whose replies come from a script,
                    so the *real* LLMReasoner prompt-building and parsing code
                    is exercised end to end.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from agent.llm_reasoner import LLMReasoner

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "is", "was", "in", "to", "and", "what", "who", "which", "question",
         "answer", "title", "by", "for", "on", "at", "as"}


def words(text: str) -> List[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class HashEncoder:
    """Deterministic bag-of-words embedding; similar texts get similar vectors."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.calls = 0

    def encode(self, texts: Sequence[str], normalize_embeddings: bool = True, **_: object) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for w in words(t):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if normalize_embeddings and n > 0:
                out[i] /= n
        return out


class FakeNLIModel:
    """Entailment if every content word of the claim's answer part occurs in the premise."""

    config = SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})

    def __init__(self) -> None:
        self.n_pairs = 0

    def predict(self, pairs, apply_softmax: bool = True, **_: object) -> np.ndarray:
        rows = []
        for premise, hypothesis in pairs:
            self.n_pairs += 1
            m = re.search(r'" is (.+)\.$', hypothesis)
            claim = m.group(1) if m else hypothesis
            claim_words = set(words(claim))
            prem = set(words(premise))
            if claim_words and claim_words <= prem:
                rows.append([0.02, 0.95, 0.03])
            elif claim_words & prem:
                ent = 0.1 + 0.6 * len(claim_words & prem) / len(claim_words)  # partial support
                rows.append([0.05, ent, 0.95 - ent])
            else:
                rows.append([0.10, 0.05, 0.85])
        return np.asarray(rows, dtype=float)


Script = Union[str, List[str]]


class FakeChatClient:
    """
    Mimics ``openai.OpenAI().chat.completions.create``.

    ``answers`` maps a question substring to a reply (or a list of replies
    consumed in order, to simulate a wrong first answer that a correction fixes).
    """

    def __init__(
        self,
        answers: Optional[Dict[str, Script]] = None,
        decompositions: Optional[Dict[str, List[str]]] = None,
        responder: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        self.answers = answers or {}
        self.decompositions = decompositions or {}
        self.responder = responder
        self.prompts: List[str] = []
        self._served: Dict[str, int] = defaultdict(int)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: object):
        prompt = kwargs["messages"][0]["content"]  # type: ignore[index]
        self.prompts.append(prompt)
        text = (self.responder(prompt) if self.responder else None) or self._default(prompt)
        usage = SimpleNamespace(prompt_tokens=len(prompt) // 4, completion_tokens=max(1, len(text) // 4))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=usage)

    def _scripted(self, question: str) -> Optional[str]:
        for key, reply in self.answers.items():
            if key.lower() in question.lower():
                if isinstance(reply, list):
                    i = self._served[key]
                    self._served[key] += 1
                    return reply[min(i, len(reply) - 1)]
                return reply
        return None

    def _default(self, prompt: str) -> str:
        if "VERDICT: SUPPORTED" in prompt:
            ans = re.search(r"\nANSWER: (.*)\n", prompt).group(1)
            ctx = prompt.split("CONTEXT:", 1)[1].split("QUESTION:", 1)[0]
            ok = set(words(ans)) <= set(words(ctx))
            return "VERDICT: SUPPORTED - found" if ok else "VERDICT: UNSUPPORTED - not in context"
        if prompt.startswith("Break this"):
            q = re.search(r"\nQuestion: (.*)\nOutput exactly", prompt).group(1)
            for key, subs in self.decompositions.items():
                if key.lower() in q.lower():
                    return "\n".join(f"{i}. {s}" for i, s in enumerate(subs, 1))
            return "1. What is the first part?\n2. What is #1 related to?"
        if "Write ONE short search query" in prompt:
            q = re.search(r"Question: (.*)\n", prompt).group(1)
            return self._scripted("SEARCH::" + q) or q
        if prompt.startswith("Rephrase this question"):
            return re.search(r"Question: (.*)\n", prompt).group(1)
        # Answer prompts: the question is the last "Q: ..." or "Question: ..." line.
        qs = re.findall(r"\n(?:Q|Question): (.*)\n", prompt)
        question = qs[-1] if qs else ""
        reply = self._scripted(question)
        if reply is None:
            reply = "I don't know"
        if prompt.rstrip().endswith("Reasoning:"):
            return f"Looking at the context.\nAnswer: {reply}"
        return reply


def make_llm(client: FakeChatClient, backend: str = "groq") -> LLMReasoner:
    """A real LLMReasoner wired to the fake client (no network, no rate limiting)."""
    return LLMReasoner(backend=backend, model="fake-model", client=client)
