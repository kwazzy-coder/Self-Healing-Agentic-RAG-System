"""
agent/cost_tracker.py — Per-query cost and latency accounting.

LLM calls and tokens are measured at the source (``LLMReasoner.usage``) and
attached per query, so the numbers cannot drift from reality the way the old
hand-placed ``record_llm_call()`` counters did.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

# The tracker for the query currently running in this thread/context. The LLM
# client reports every call here, so concurrent API requests never mix counts.
CURRENT_TRACKER: ContextVar[Optional[CostTracker]] = ContextVar("current_cost_tracker", default=None)


@dataclass
class QueryCost:
    n_llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_retrieval_calls: int = 0
    n_nli_pairs: int = 0
    monitor_tiers: List[int] = field(default_factory=list)
    correction_rounds: int = 0
    component_latencies: Dict[str, float] = field(default_factory=dict)
    total_latency_seconds: float = 0.0

    @property
    def monitor_tier_used(self) -> int:
        """Highest tier reached on any monitor call for this query."""
        return max(self.monitor_tiers, default=0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_llm_calls": self.n_llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "n_retrieval_calls": self.n_retrieval_calls,
            "n_nli_pairs": self.n_nli_pairs,
            "monitor_tiers": list(self.monitor_tiers),
            "monitor_tier_used": self.monitor_tier_used,
            "correction_rounds": self.correction_rounds,
            "component_latencies": {k: round(v, 4) for k, v in self.component_latencies.items()},
            "total_latency_seconds": round(self.total_latency_seconds, 4),
        }


class CostTracker:
    """Collects QueryCost objects. One tracker per pipeline *call* (thread-safe by construction)."""

    def __init__(self) -> None:
        self.current = QueryCost()
        self._start = time.perf_counter()

    # Backwards-compatible no-op (a fresh tracker is created per query now).
    def reset_query(self) -> None:
        self.current = QueryCost()
        self._start = time.perf_counter()

    def record_retrieval_call(self) -> None:
        self.current.n_retrieval_calls += 1

    def record_nli_inference(self, n_pairs: int = 1) -> None:
        self.current.n_nli_pairs += n_pairs

    def record_monitor_tier(self, tier: int) -> None:
        self.current.monitor_tiers.append(int(tier))

    def record_correction_round(self) -> None:
        self.current.correction_rounds += 1

    def record_llm_usage(self, calls: int, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.current.n_llm_calls += calls
        self.current.prompt_tokens += prompt_tokens
        self.current.completion_tokens += completion_tokens

    @contextmanager
    def time_component(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            lat = self.current.component_latencies
            lat[name] = lat.get(name, 0.0) + (time.perf_counter() - start)

    def finalize_query(self) -> QueryCost:
        self.current.total_latency_seconds = time.perf_counter() - self._start
        return self.current


def aggregate_costs(costs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Average a list of QueryCost.to_dict() records (used by the evaluator)."""
    n = len(costs)
    if n == 0:
        return {}
    tiers = [c.get("monitor_tier_used", 0) for c in costs]

    def avg(key: str) -> float:
        return sum(float(c.get(key, 0) or 0) for c in costs) / n

    return {
        "n_queries": n,
        "avg_llm_calls": avg("n_llm_calls"),
        "avg_prompt_tokens": avg("prompt_tokens"),
        "avg_completion_tokens": avg("completion_tokens"),
        "avg_retrieval_calls": avg("n_retrieval_calls"),
        "avg_nli_pairs": avg("n_nli_pairs"),
        "avg_correction_rounds": avg("correction_rounds"),
        "avg_latency_seconds": avg("total_latency_seconds"),
        "tier_distribution": {f"tier_{t}": tiers.count(t) / n for t in range(4)},
        "total_llm_calls": int(sum(c.get("n_llm_calls", 0) for c in costs)),
    }
