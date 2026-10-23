"""
agent/nli.py — Natural Language Inference scorer for grounding checks.

Uses a small cross-encoder (cross-encoder/nli-deberta-v3-small, ~70M params)
to estimate P(entailment) of a hypothesis (the answer, as a claim) given
premises (the retrieved passages).

Fixes over the original Tier-2 check:
* premise and hypothesis are passed as a proper *pair* (the original glued
  them into one string with a RoBERTa-style separator, which DeBERTa ignores);
* the full softmax distribution is used, not just the top label;
* every passage is scored separately (plus one joint premise of the top
  passages, so two-hop evidence can still entail the answer) instead of one
  2,000-character blob that cut off most of the context;
* short answers are turned into a claim using the question, instead of
  getting an automatic pass (which skipped almost every HotpotQA answer).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

import config
from retrieval.chunking import split_sentences

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict = {}
_LOCK = threading.Lock()


@dataclass
class NLIScore:
    hypothesis: str
    entailment: float
    contradiction: float
    best_premise: int


def answer_to_hypotheses(question: str, answer: str, short_max_words: int = 12) -> List[str]:
    """
    Turn an answer into NLI hypotheses.

    Short answers ("Paris") become one claim built from the question;
    long answers are checked sentence by sentence.
    """
    answer = (answer or "").strip()
    if not answer:
        return []
    if len(answer.split()) <= short_max_words:
        return [f'The answer to the question "{question.strip()}" is {answer.rstrip(".")}.']
    sents = [s for s in split_sentences(answer) if len(s.split()) >= 3]
    return sents or [answer]


def build_premises(passages: Sequence[str], max_chars: int, joint_top: int = 2) -> List[str]:
    """Split passages into NLI-sized windows and add one joint premise of the top passages."""
    premises: List[str] = []
    for p in passages:
        p = (p or "").strip()
        if not p:
            continue
        if len(p) <= max_chars:
            premises.append(p)
            continue
        window = ""
        for sent in split_sentences(p):
            if window and len(window) + len(sent) + 1 > max_chars:
                premises.append(window)
                window = ""
            window = f"{window} {sent}".strip()
        if window:
            premises.append(window)
    top = [p.strip() for p in passages[:joint_top] if p and p.strip()]
    if len(top) >= 2:
        per = max(200, (max_chars * 2) // len(top))
        premises.append(" ".join(t[:per] for t in top))
    return premises


class NLIScorer:
    """Thin wrapper around a sentence-transformers CrossEncoder NLI model."""

    def __init__(self, model_name: Optional[str] = None, model: Any = None) -> None:
        self.model_name = model_name or config.NLI_MODEL
        self._model = model
        self._label_idx: Optional[dict] = None

    @property
    def model(self) -> Any:
        if self._model is None:
            with _LOCK:
                if self.model_name not in _MODEL_CACHE:
                    from sentence_transformers import CrossEncoder

                    logger.info("Loading NLI model: %s", self.model_name)
                    _MODEL_CACHE[self.model_name] = CrossEncoder(self.model_name)
                self._model = _MODEL_CACHE[self.model_name]
        return self._model

    def _labels(self) -> dict:
        if self._label_idx is None:
            id2label = None
            for holder in (self.model, getattr(self.model, "model", None)):
                cfg = getattr(holder, "config", None)
                if cfg is not None and getattr(cfg, "id2label", None):
                    id2label = cfg.id2label
                    break
            # Default label order of cross-encoder/nli-* models.
            id2label = id2label or {0: "contradiction", 1: "entailment", 2: "neutral"}
            lookup = {str(v).lower(): int(k) for k, v in id2label.items()}
            self._label_idx = {
                "entailment": lookup.get("entailment", 1),
                "contradiction": lookup.get("contradiction", 0),
            }
        return self._label_idx

    def score(self, premises: Sequence[str], hypotheses: Sequence[str]) -> List[NLIScore]:
        """For each hypothesis, max P(entail) and max P(contradiction) over all premises."""
        if not premises or not hypotheses:
            return [NLIScore(h, 0.0, 0.0, -1) for h in hypotheses]
        pairs = [(p, h) for h in hypotheses for p in premises]
        probs = np.asarray(self.model.predict(pairs, apply_softmax=True, show_progress_bar=False), dtype=float)
        probs = probs.reshape(len(hypotheses), len(premises), -1)
        idx = self._labels()
        out = []
        for h_i, hyp in enumerate(hypotheses):
            ent = probs[h_i, :, idx["entailment"]]
            con = probs[h_i, :, idx["contradiction"]]
            best = int(np.argmax(ent))
            out.append(NLIScore(hyp, float(ent[best]), float(con.max()), best))
        return out
