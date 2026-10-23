"""
api/server.py — FastAPI service.

    uvicorn api.server:app --host 0.0.0.0 --port 8000
    open http://localhost:8000/docs   (interactive Swagger UI)

Endpoints
  GET  /health      liveness + configured provider/model
  POST /v1/ask      answer a question over the supplied documents (with self-healing)
  POST /v1/verify   hallucination check only: is this answer grounded in these documents?

Endpoints are plain ``def`` functions, so FastAPI runs them in a thread pool;
the pipeline keeps no per-request state, so concurrent requests are safe.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

import config

logger = logging.getLogger(__name__)

MAX_DOCS = 50
MAX_DOC_CHARS = 20_000

app = FastAPI(
    title="Self-Healing Agentic RAG",
    version="2.0.0",
    description="RAG that detects ungrounded answers (heuristics → NLI → LLM judge) and repairs them.",
)

_pipeline = None
_lock = threading.Lock()


def get_pipeline() -> Any:
    """Build the pipeline once, on first request (tests override this dependency)."""
    global _pipeline
    with _lock:
        if _pipeline is None:
            from pipeline import SelfHealingRAGPipeline

            _pipeline = SelfHealingRAGPipeline()
    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, examples=["Who directed the film that stars X?"])
    documents: List[str] = Field(..., min_length=1, max_length=MAX_DOCS,
                                 description="Passages to search. Prefix with 'Title: <name>\\n' to keep titles.")
    question_type: Optional[Literal["bridge", "comparison", "direct"]] = None
    answer_style: Literal["short", "long"] = "short"
    max_correction_rounds: Optional[int] = Field(None, ge=0, le=4)
    include_trace: bool = True


class RoundOut(BaseModel):
    round: int
    produced_by: str
    answer: str
    has_failure: Optional[bool]
    failure_type: str = ""
    monitor_tier: int = 0
    confidence: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    initial_answer: str
    healed: bool = Field(description="True if self-healing replaced the initial answer")
    verified: Optional[bool] = Field(description="True if the returned answer passed the hallucination monitor")
    failure_detected: bool
    failure_type: str
    strategy: str
    rounds: List[RoundOut]
    sources: List[str]
    cost: Dict[str, Any]
    latency_ms: float
    trace: Optional[List[str]] = None


class VerifyRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    answer: str = Field(..., min_length=1, max_length=5000)
    documents: List[str] = Field(..., min_length=1, max_length=MAX_DOCS)
    mode: Literal["tiered", "heuristic", "nli", "heuristic_nli", "llm"] = "tiered"


class VerifyResponse(BaseModel):
    grounded: bool
    failure_type: str
    reason: str
    tier: int
    confidence: float
    nli_entailment: Optional[float] = None
    judge_verdict: Optional[bool] = None


def _check_docs(docs: List[str]) -> None:
    too_long = [i for i, d in enumerate(docs) if len(d) > MAX_DOC_CHARS]
    if too_long:
        raise HTTPException(422, f"documents {too_long} exceed {MAX_DOC_CHARS} characters")


@app.get("/health")
def health() -> Dict[str, Any]:
    if _pipeline is not None:  # report what is actually in use (after any provider fallback)
        return {"status": "ok", "provider": _pipeline.llm.backend, "model": _pipeline.llm.model,
                "pipeline_loaded": True}
    return {"status": "ok", "provider": config.LLM_BACKEND,
            "model": config.default_model_for(config.LLM_BACKEND), "pipeline_loaded": False}


@app.post("/v1/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline: Any = Depends(get_pipeline)) -> AskResponse:
    from agent.llm_reasoner import LLMError
    from pipeline import PipelineConfig

    _check_docs(req.documents)
    cfg = PipelineConfig(
        name="api", short_answers=req.answer_style == "short", chunk_size=1000,
        max_correction_rounds=req.max_correction_rounds,
    )
    qt = req.question_type  # None = auto-detect; "direct" forces single-hop answering
    start = time.perf_counter()
    try:
        out = pipeline.run(req.question, req.documents, question_type=qt, pipeline_config=cfg)
    except LLMError as err:
        raise HTTPException(502, f"LLM provider error: {err}") from err
    return AskResponse(
        answer=out.final_answer,
        initial_answer=out.initial_answer,
        healed=out.selected_round > 0,
        verified=out.verified,
        failure_detected=out.failure_detected,
        failure_type=out.failure_type,
        strategy=out.strategy,
        rounds=[RoundOut(**{k: getattr(r, k) for k in RoundOut.model_fields}) for r in out.rounds],
        sources=out.retrieved_docs,
        cost=out.cost_data,
        latency_ms=round(1000 * (time.perf_counter() - start), 1),
        trace=out.trace if req.include_trace else None,
    )


@app.post("/v1/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest, pipeline: Any = Depends(get_pipeline)) -> VerifyResponse:
    from agent.self_monitor import MonitorSettings, SelfMonitor

    _check_docs(req.documents)
    short = len(req.answer.split()) <= 12
    monitor = SelfMonitor(pipeline.llm, nli=pipeline.components.nli,
                          settings=MonitorSettings.from_mode(req.mode, short_answers=short))
    report = monitor.evaluate(req.question, req.answer, req.documents)
    return VerifyResponse(
        grounded=not report.has_failure,
        failure_type=report.failure_type,
        reason=report.failure_reason,
        tier=report.monitor_tier_used,
        confidence=round(float(report.confidence_score), 4),
        nli_entailment=report.nli_entailment,
        judge_verdict=report.judge_verdict,
    )
