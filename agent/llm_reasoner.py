"""
agent/llm_reasoner.py — Provider-agnostic LLM interface.

Design
------
* Gemini, Groq, OpenAI and Ollama all expose an OpenAI-compatible Chat
  Completions API, so a single client (``openai`` SDK + ``base_url``) covers
  all four. Switching provider is an env-var change, not a code change.
* HuggingFace Flan-T5 runs locally as a no-key fallback.
* Retries with exponential backoff (honouring ``Retry-After``) on 429/5xx,
  a shared per-provider rate limiter for free tiers, and token accounting.
* Errors are raised as ``LLMError`` instead of being silently converted into
  "I don't know" (which used to look like a model refusal and polluted metrics).
"""
from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Sequence, Tuple

import config
from agent import prompts
from agent.cost_tracker import CURRENT_TRACKER

logger = logging.getLogger(__name__)

API_PROVIDERS = ("gemini", "groq", "openai", "ollama")
FALLBACK_ORDER = ("gemini", "groq", "openai", "ollama", "huggingface")
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class LLMError(RuntimeError):
    """Raised when the LLM provider fails after all retries."""


@dataclass
class LLMUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0

    def snapshot(self) -> LLMUsage:
        return LLMUsage(**asdict(self))

    def minus(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            calls=self.calls - other.calls,
            prompt_tokens=self.prompt_tokens - other.prompt_tokens,
            completion_tokens=self.completion_tokens - other.completion_tokens,
            retries=self.retries - other.retries,
        )


class _RateLimiter:
    """Thread-safe minimum-gap rate limiter (shared per provider)."""

    def __init__(self, min_delay: float) -> None:
        self._lock = threading.Lock()
        self._last = 0.0
        self.min_delay = min_delay

    def wait(self) -> None:
        if self.min_delay <= 0:
            return
        with self._lock:
            gap = self.min_delay - (time.monotonic() - self._last)
            if gap > 0 and self._last > 0:
                time.sleep(gap)
            self._last = time.monotonic()


_LIMITERS: dict = {}
_LIMITERS_LOCK = threading.Lock()


def _limiter_for(provider: str) -> _RateLimiter:
    with _LIMITERS_LOCK:
        if provider not in _LIMITERS:
            _LIMITERS[provider] = _RateLimiter(config.min_delay_for(provider))
        return _LIMITERS[provider]


def _has_key(provider: str) -> bool:
    if provider == "gemini":
        return bool(config.GEMINI_API_KEY)
    if provider == "groq":
        return bool(config.GROQ_API_KEY)
    if provider == "openai":
        return bool(config.OPENAI_API_KEY)
    if provider == "ollama":
        return _ollama_available()
    return provider == "huggingface"


def _ollama_available() -> bool:
    base = config.PROVIDER_BASE_URLS["ollama"].rsplit("/v1", 1)[0]
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=2) as resp:  # noqa: S310
            models = [m.get("name", "") for m in json.load(resp).get("models", [])]
        wanted = config.default_model_for("ollama").split(":")[0]
        return any(wanted in m for m in models)
    except Exception:
        return False


def resolve_provider(requested: Optional[str]) -> str:
    """Return the requested provider if usable, else the first usable fallback."""
    requested = (requested or config.LLM_BACKEND or "gemini").lower()
    if requested == "huggingface" or _has_key(requested):
        return requested
    for candidate in FALLBACK_ORDER:
        if candidate != requested and _has_key(candidate):
            logger.warning(
                "LLM provider '%s' is not configured (missing key/server); falling back to '%s'.",
                requested,
                candidate,
            )
            return candidate
    return "huggingface"


class LLMReasoner:
    """Generates answers, judgements and sub-questions with an LLM."""

    def __init__(
        self,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        client: Any = None,
    ) -> None:
        """
        Args:
            backend: Provider name (gemini/groq/openai/ollama/huggingface).
            model: Model id; defaults to the provider default in config.
            client: Optional pre-built OpenAI-compatible client (used in tests).
        """
        self.backend = backend.lower() if (backend and client is not None) else resolve_provider(backend)
        self.model = model or config.default_model_for(self.backend)
        self.usage = LLMUsage()
        self._usage_lock = threading.Lock()
        self._client = client
        self._send_temperature = True
        self._hf_tokenizer = None
        self._hf_model = None
        if self._client is None:
            self._init_client()
        logger.info("LLM ready: provider=%s model=%s", self.backend, self.model)

    # ── Setup ────────────────────────────────────────────────────────────────
    @property
    def is_api(self) -> bool:
        """True for capable instruction-following API models (not local Flan-T5)."""
        return self.backend in API_PROVIDERS

    def _init_client(self) -> None:
        if self.backend == "huggingface":
            self._init_hf()
            return
        from openai import OpenAI

        key = {
            "gemini": config.GEMINI_API_KEY,
            "groq": config.GROQ_API_KEY,
            "openai": config.OPENAI_API_KEY,
            "ollama": "ollama",  # Ollama ignores the key but the SDK requires one
        }[self.backend]
        kwargs = {"api_key": key, "timeout": config.REQUEST_TIMEOUT_S, "max_retries": 0}
        base_url = config.PROVIDER_BASE_URLS.get(self.backend)
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def _init_hf(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading local model %s (first run downloads ~1GB)...", self.model)
        self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._hf_model = AutoModelForSeq2SeqLM.from_pretrained(self.model)

    # ── Core generation ──────────────────────────────────────────────────────
    def generate(self, prompt: str, max_tokens: int = 256, temperature: Optional[float] = None) -> str:
        """Generate text for a prompt. Raises LLMError after exhausting retries."""
        temperature = config.TEMPERATURE if temperature is None else temperature
        if self.backend == "huggingface":
            text = self._hf_generate(prompt, max_tokens)
            self._record(prompt_tokens=0, completion_tokens=0)
            return text

        limiter = _limiter_for(self.backend)
        last_err: Optional[Exception] = None
        for attempt in range(config.MAX_RETRIES + 1):
            limiter.wait()
            try:
                return self._api_generate(prompt, max_tokens, temperature)
            except Exception as err:  # noqa: BLE001 - classify below
                last_err = err
                if self._is_temperature_error(err) and self._send_temperature:
                    logger.info("Model %s rejects 'temperature'; retrying without it.", self.model)
                    self._send_temperature = False
                    continue
                if not self._is_retryable(err) or attempt == config.MAX_RETRIES:
                    break
                wait = self._backoff_seconds(err, attempt)
                with self._usage_lock:
                    self.usage.retries += 1
                logger.warning("LLM call failed (%s). Retrying in %.1fs...", type(err).__name__, wait)
                time.sleep(wait)
        raise LLMError(f"{self.backend}/{self.model} failed: {last_err}") from last_err

    def _api_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        budget = max(max_tokens, config.MIN_OUTPUT_TOKENS)
        kwargs: dict = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        # Newer parameters go through extra_body so that older openai SDK versions work too.
        extra: dict = {}
        if self.backend in ("openai", "groq"):  # reasoning models want max_completion_tokens
            extra["max_completion_tokens"] = budget
        else:
            kwargs["max_tokens"] = budget
        if self._send_temperature:
            kwargs["temperature"] = temperature
        if config.LLM_REASONING_EFFORT and self.backend in ("gemini", "groq", "openai"):
            extra["reasoning_effort"] = config.LLM_REASONING_EFFORT
        if extra:
            kwargs["extra_body"] = extra

        resp = self._client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        if not text and resp.choices and getattr(resp.choices[0], "finish_reason", None) == "length":
            logger.warning("Empty completion (output budget used up by reasoning). "
                           "Raise MIN_OUTPUT_TOKENS or lower LLM_REASONING_EFFORT.")
        usage = getattr(resp, "usage", None)
        self._record(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        return text

    def _hf_generate(self, prompt: str, max_tokens: int) -> str:
        import torch

        inputs = self._hf_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = self._hf_model.generate(
                **inputs, max_new_tokens=min(max_tokens, 64), do_sample=False, num_beams=4, early_stopping=True
            )
        return self._hf_tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def _record(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._usage_lock:
            self.usage.calls += 1
            self.usage.prompt_tokens += int(prompt_tokens)
            self.usage.completion_tokens += int(completion_tokens)
        tracker = CURRENT_TRACKER.get()
        if tracker is not None:
            tracker.record_llm_usage(1, int(prompt_tokens), int(completion_tokens))

    @staticmethod
    def _status(err: Exception) -> Optional[int]:
        return getattr(err, "status_code", None) or getattr(getattr(err, "response", None), "status_code", None)

    def _is_retryable(self, err: Exception) -> bool:
        name = type(err).__name__
        if name in ("RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"):
            return True
        status = self._status(err)
        return status in RETRYABLE_STATUS

    def _is_temperature_error(self, err: Exception) -> bool:
        return self._status(err) == 400 and "temperature" in str(err).lower()

    def _backoff_seconds(self, err: Exception, attempt: int) -> float:
        headers = getattr(getattr(err, "response", None), "headers", None) or {}
        retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
        if retry_after:
            try:
                return min(float(retry_after) + 1.0, config.RETRY_MAX_WAIT_S)
            except ValueError:
                pass
        base = config.RETRY_BASE_WAIT_S * (2 ** attempt)
        return min(base, config.RETRY_MAX_WAIT_S) + random.uniform(0, 1)

    # ── Task-level helpers ───────────────────────────────────────────────────
    def _context(self, passages: Sequence[str], reserve: int = 0) -> str:
        limit = config.CONTEXT_CHAR_LIMIT if self.is_api else config.HF_CONTEXT_CHAR_LIMIT
        return prompts.format_passages(passages, max(500, limit - reserve))

    def answer_with_context(self, question: str, passages: Sequence[str], short: bool = True) -> str:
        """Answer a question from passages (Chain-of-Thought for short answers)."""
        if not self.is_api:
            context = self._context(passages)
            return self._clean(self.generate(f"Context: {context}\n\nQuestion: {question}\nAnswer:", 32))
        context = self._context(passages)
        if short:
            raw = self.generate(prompts.short_answer_prompt(question, context), max_tokens=256)
            return self.extract_final_answer(raw)
        return self.generate(prompts.long_answer_prompt(question, context), max_tokens=400).strip()

    def answer_closed_book(self, question: str, short: bool = True) -> str:
        """No-retrieval baseline: answer from parametric knowledge only."""
        if not self.is_api:
            return self._clean(self.generate(f"Question: {question}\nAnswer:", 32))
        raw = self.generate(prompts.closed_book_prompt(question, short), max_tokens=256)
        return self.extract_final_answer(raw) if short else raw.strip()

    def answer_with_sub_answers(
        self,
        question: str,
        sub_qas: Sequence[Tuple[str, str]],
        passages: Sequence[str],
        short: bool = True,
    ) -> str:
        """Synthesize a final answer from (sub-question, sub-answer) pairs + context."""
        sub_qas = [(q, a) for q, a in sub_qas]
        if not self.is_api:
            hints = " | ".join(a for _, a in sub_qas)
            context = self._context(passages, reserve=200)
            return self._clean(self.generate(f"Context: {context} {hints}\n\nQuestion: {question}\nAnswer:", 32))
        context = self._context(passages, reserve=600)
        raw = self.generate(prompts.synthesis_prompt(question, sub_qas, context, short=short), max_tokens=300)
        return self.extract_final_answer(raw) if short else raw.strip()

    def decompose(self, question: str, strategy: str) -> List[str]:
        """Split a multi-hop question into (up to) 2 sub-questions."""
        raw = self.generate(prompts.decompose_prompt(question, strategy), max_tokens=120)
        return parse_numbered_lines(raw)[:2]

    def judge_support(self, question: str, answer: str, passages: Sequence[str]) -> Tuple[Optional[bool], str]:
        """
        LLM-as-judge grounding verdict.

        Returns (True|False|None, reason). None means the reply could not be parsed.
        """
        raw = self.generate(prompts.judge_prompt(question, answer, self._context(passages)), max_tokens=80)
        return parse_verdict(raw)

    def search_query(self, question: str, previous_answer: str, unsupported: Sequence[str]) -> str:
        """Ask the LLM for a targeted follow-up search query."""
        raw = self.generate(prompts.search_query_prompt(question, previous_answer, unsupported), max_tokens=40)
        query = raw.strip().splitlines()[0].strip().strip('"\'') if raw.strip() else ""
        return query if len(query) >= 4 else question

    def rephrase(self, question: str) -> str:
        raw = self.generate(prompts.rephrase_prompt(question), max_tokens=60)
        new_q = raw.strip().splitlines()[0].strip().strip('"\'') if raw.strip() else ""
        return new_q if len(new_q) >= 5 else question

    # ── Output parsing ───────────────────────────────────────────────────────
    def extract_final_answer(self, raw: str) -> str:
        """Extract the text after the last 'Answer:' marker of a CoT completion."""
        if not raw:
            return ""
        parts = re.split(r"answer\s*:", raw, flags=re.IGNORECASE)
        if len(parts) > 1:
            candidate = parts[-1].strip().split("\n")[0].strip()
            if candidate:
                return self._clean(candidate)
        lines = [ln.strip() for ln in raw.strip().split("\n") if ln.strip()]
        return self._clean(lines[-1] if lines else raw)

    # Kept for backwards compatibility with older callers.
    _extract_final_answer = extract_final_answer

    def _clean(self, answer: str) -> str:
        """Normalise a short answer (strip boilerplate, keep it concise)."""
        if not answer:
            return ""
        a = answer.strip().strip("*").strip().strip("\"'")
        for pat in (r"^the answer is[:\s]+", r"^answer[:\s]+", r"^so the answer is[:\s]+",
                    r"^therefore[,\s]+", r"^thus[,\s]+"):
            a = re.sub(pat, "", a, flags=re.IGNORECASE).strip()

        yes_no = re.match(r"^(yes|no)\b(.*)$", a, re.IGNORECASE | re.DOTALL)
        if yes_no:
            rest = yes_no.group(2).strip()
            # "Yes, both are..." / "no." / "yes they are" -> yes/no, but keep "No Doubt" (a band).
            if not rest or rest[0] in ",.!;:-" or rest[:1].islower():
                return yes_no.group(1).lower()

        match = re.match(r"^(.+?)\s+(is|was|are|were)\s+(the|a|an)\s+", a, re.IGNORECASE)
        if match and 1 <= len(match.group(1).split()) <= 5:
            a = match.group(1).strip()

        a = re.sub(r"\s*\(.*?\)", "", a)
        a = a.strip().rstrip(".,;:\"'")
        words = a.split()
        if len(words) > config.MAX_ANSWER_WORDS + 1:
            a = " ".join(words[: config.MAX_ANSWER_WORDS])
        return a.strip()


def parse_numbered_lines(raw: str) -> List[str]:
    """Parse '1. foo\\n2. bar' (or '1) foo') into ['foo', 'bar']."""
    out = []
    for line in (raw or "").splitlines():
        line = line.strip()
        m = re.match(r"^(?:\*\*)?\d+[\.\):]\s*(?:\*\*)?\s*(.+)$", line)
        if m:
            text = m.group(1).strip().strip("*").strip()
            if len(text) > 5:
                out.append(text)
    return out


def parse_verdict(raw: str) -> Tuple[Optional[bool], str]:
    """Parse 'VERDICT: SUPPORTED - reason' style replies (robust to YES/NO)."""
    text = (raw or "").strip()
    upper = text.upper()
    reason = text.split("-", 1)[1].strip() if "-" in text else text
    if "UNSUPPORTED" in upper or "NOT SUPPORTED" in upper:
        return False, reason
    if "SUPPORTED" in upper:
        return True, reason
    if re.match(r"^\W*NO\b", upper):
        return False, reason
    if re.match(r"^\W*YES\b", upper):
        return True, reason
    return None, text[:200]
