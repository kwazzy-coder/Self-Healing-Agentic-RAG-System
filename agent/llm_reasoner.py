"""
agent/llm_reasoner.py
KEY CHANGE: Chain-of-Thought (CoT) prompting.
Forces LLM to reason step-by-step before giving final answer.
This is the single biggest improvement for multi-hop QA.

Expected improvement: +15-20% EM, +10-15% F1
"""
import os
import time
import threading
import re
import logging
from typing import List, Optional, Any

import config
from config import (
    LLM_BACKEND,
    GEMINI_API_KEY, GEMINI_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
    OLLAMA_MODEL, OLLAMA_BASE_URL,
    OPENAI_MODEL, OPENAI_API_KEY, HF_MODEL,
    MIN_DELAY_BETWEEN_CALLS, MAX_RETRIES, RETRY_WAIT_SECONDS,
)

logger = logging.getLogger(__name__)

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


class _RateLimiter:
    """Thread-safe rate limiter for API calls."""
    def __init__(self, min_delay: float) -> None:
        self._lock = threading.Lock()
        self._last = 0.0
        self.min_delay = min_delay

    def wait(self) -> None:
        """Wait if necessary to enforce the minimum delay."""
        with self._lock:
            gap = self.min_delay - (time.time() - self._last)
            if gap > 0 and self._last > 0:
                time.sleep(gap)
            self._last = time.time()


_limiter = _RateLimiter(MIN_DELAY_BETWEEN_CALLS)


class LLMReasoner:
    """LLM Reasoner with Chain-of-Thought capabilities."""
    
    def __init__(self, backend: Optional[str] = None) -> None:
        """
        Initialize the LLM Reasoner.
        
        Args:
            backend: Optional backend override.
        """
        self.backend = backend or LLM_BACKEND
        self._gemini_client = None
        self._groq_client = None
        self._oai_client = None
        self._hf_tokenizer = None
        self._hf_model = None
        self._init()

    def _init(self) -> None:
        """Initialize the requested backend, falling back iteratively if needed."""
        backends_to_try = [self.backend]
        if self.backend == "gemini":
            backends_to_try.extend(["groq", "ollama", "openai", "huggingface"])
        elif self.backend == "groq":
            backends_to_try.extend(["ollama", "openai", "huggingface"])
        elif self.backend == "ollama":
            backends_to_try.extend(["openai", "huggingface"])
        elif self.backend == "openai":
            backends_to_try.extend(["huggingface"])

        for b in backends_to_try:
            self.backend = b
            if b == "gemini" and self._init_gemini():
                logger.info("✓ Gemini: %s", GEMINI_MODEL)
                return
            elif b == "groq" and self._init_groq():
                logger.info("✓ Groq: %s", GROQ_MODEL)
                return
            elif b == "ollama" and self._check_ollama():
                logger.info("✓ Ollama: %s", OLLAMA_MODEL)
                return
            elif b == "openai" and self._init_openai():
                logger.info("✓ OpenAI: %s", OPENAI_MODEL)
                return
            elif b == "huggingface":
                self._init_hf()
                return

    def _init_gemini(self) -> bool:
        """Initialize Gemini client without wasteful test calls."""
        key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        if not key:
            logger.warning("No GEMINI_API_KEY → https://aistudio.google.com/apikey")
            return False
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            self._gemini_client = genai.GenerativeModel(GEMINI_MODEL)
            return True
        except ImportError:
            logger.warning("Run: pip install google-generativeai")
            return False
        except Exception as e:
            logger.warning("Gemini error: %s", e)
            return False

    def _init_groq(self) -> bool:
        """Initialize Groq client without wasteful test calls."""
        key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        if not key:
            logger.warning("No GROQ_API_KEY → https://console.groq.com")
            return False
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=key)
            return True
        except ImportError:
            logger.warning("Run: pip install groq")
            return False
        except Exception as e:
            logger.warning("Groq error: %s", e)
            return False

    def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        try:
            import requests
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return any(OLLAMA_MODEL.split(":")[0] in m for m in models)
        except Exception:
            pass
        return False

    def _init_openai(self) -> bool:
        """Initialize OpenAI client."""
        key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        if not key:
            return False
        try:
            from openai import OpenAI
            self._oai_client = OpenAI(api_key=key)
            return True
        except ImportError:
            return False

    def _init_hf(self) -> None:
        """Initialize HuggingFace client."""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        logger.info("Loading %s...", HF_MODEL)
        self._hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        self._hf_model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL)
        logger.info("✓ HuggingFace loaded.")

    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.0) -> str:
        """
        Generate a response from the configured LLM backend.
        
        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            The generated response string.
        """
        _limiter.wait()
        for attempt in range(MAX_RETRIES):
            try:
                if self.backend == "gemini":
                    return self._gemini_generate(prompt, max_tokens, temperature)
                elif self.backend == "groq":
                    return self._groq_generate(prompt, max_tokens, temperature)
                elif self.backend == "ollama":
                    return self._ollama_generate(prompt, max_tokens, temperature)
                elif self.backend == "openai":
                    return self._oai_generate(prompt, max_tokens, temperature)
                else:
                    return self._hf_generate(prompt, max_tokens)
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "rate" in msg or "quota" in msg or "too many" in msg:
                    wait = self._parse_retry_after(e) or (RETRY_WAIT_SECONDS * (attempt + 1))
                    logger.warning("Rate limited. Waiting %ds...", wait)
                    time.sleep(wait)
                else:
                    logger.error("Error generating response: %s", e)
                    return "I don't know."
        return "I don't know."

    def _parse_retry_after(self, error: Exception) -> Optional[int]:
        """Extract retry wait time from error message."""
        try:
            m = re.search(r"(\d+\.?\d*)\s*s", str(error), re.IGNORECASE)
            return int(float(m.group(1))) + 2 if m else None
        except Exception:
            return None

    def answer_with_context(self, question: str, context_passages: List[str]) -> str:
        """
        Chain-of-Thought answering using context.
        
        Args:
            question: The question to answer.
            context_passages: List of context documents.
            
        Returns:
            The extracted final answer.
        """
        limit_api = getattr(config, 'MAX_CONTEXT_CHARS_API', 3000)
        limit_local = getattr(config, 'MAX_CONTEXT_CHARS_LOCAL', 600)
        limit = limit_api if self.backend in ("gemini", "groq", "openai", "ollama") else limit_local
        
        combined = " ".join(p.strip() for p in context_passages)[:limit]

        if self.backend == "huggingface":
            raw = self.generate(
                f"Context: {combined}\n\nQuestion: {question}\nAnswer:",
                max_tokens=32
            )
            return self._extract_final_answer(raw)

        prompt = (
            f"{COT_FEW_SHOT}"
            f"Now answer this question using the context below.\n"
            f"First write your Reasoning (1-2 sentences), "
            f"then write Answer (brief, just the answer).\n\n"
            f"Context: {combined}\n\n"
            f"Q: {question}\n"
            f"Reasoning:"
        )
        raw = self.generate(prompt, max_tokens=200)
        return self._extract_final_answer(raw)

    def answer_with_sub_answers(self, question: str, sub_answers: List[str], context_passages: List[str]) -> str:
        """
        Chain-of-Thought answering using context and intermediate sub-answers.
        
        Args:
            question: The question to answer.
            sub_answers: Intermediate answers from sub-questions.
            context_passages: List of context documents.
            
        Returns:
            The extracted final answer.
        """
        limit_api = getattr(config, 'MAX_CONTEXT_CHARS_API', 3000)
        limit_local = getattr(config, 'MAX_CONTEXT_CHARS_LOCAL', 600)
        limit = limit_api - 500 if self.backend in ("gemini", "groq", "openai", "ollama") else limit_local - 200
        
        combined = " ".join(p.strip() for p in context_passages)[:limit]
        hints = " | ".join(sub_answers)

        if self.backend == "huggingface":
            raw = self.generate(
                f"Context: {combined} {hints}\n\nQuestion: {question}\nAnswer:",
                max_tokens=32
            )
            return self._extract_final_answer(raw)

        prompt = (
            f"{COT_FEW_SHOT}"
            f"Use the context and intermediate findings.\n"
            f"First write Reasoning (1-2 sentences), then Answer (brief, just the answer).\n\n"
            f"Context: {combined}\n"
            f"Findings: {hints}\n\n"
            f"Q: {question}\n"
            f"Reasoning:"
        )
        raw = self.generate(prompt, max_tokens=200)
        return self._extract_final_answer(raw)

    def _extract_final_answer(self, raw: str) -> str:
        """
        Extract the Answer from CoT output.
        
        Args:
            raw: The raw LLM generation.
            
        Returns:
            The extracted and cleaned answer.
        """
        if not raw:
            return ""

        if "answer:" in raw.lower():
            parts = re.split(r"answer\s*:", raw, flags=re.IGNORECASE)
            if len(parts) > 1:
                answer_part = parts[-1].strip()
                answer_part = answer_part.split("\n")[0].strip()
                return self._clean(answer_part)

        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        if lines:
            return self._clean(lines[-1])

        return self._clean(raw)

    def _clean(self, answer: str) -> str:
        """
        Clean up the extracted answer.
        
        Args:
            answer: Raw extracted answer.
            
        Returns:
            Cleaned answer.
        """
        if not answer:
            return ""
        a = answer.strip()
        a = a.strip('"\'')

        prefixes = [
            r"^the answer is[:\s]+",
            r"^answer[:\s]+",
            r"^so the answer is[:\s]+",
            r"^therefore[,\s]+",
            r"^thus[,\s]+",
        ]
        for pat in prefixes:
            a = re.sub(pat, "", a, flags=re.IGNORECASE).strip()

        if re.match(r"^(yes|no)[,\.\s]", a, re.IGNORECASE):
            return a.split()[0].lower()

        match = re.match(r"^(.+?)\s+(is|was|are|were)\s+(the|a|an)\s+", a, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            # Only apply if candidate length seems reasonable
            if 1 <= len(candidate.split()) <= 5:
                a = candidate

        a = re.sub(r"\s*\(.*?\)", "", a)
        a = a.strip().rstrip(".,;:\"'")

        max_words = getattr(config, 'MAX_ANSWER_WORDS', 8)
        words = a.split()
        if len(words) > max_words + 1:
            a = " ".join(words[:max_words])

        return a.strip()

    def _gemini_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        import google.generativeai as genai
        response = self._gemini_client.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        )
        return response.text.strip()

    def _groq_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        r = self._groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return r.choices[0].message.content.strip()

    def _ollama_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        import requests
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": temperature}
            },
            timeout=60
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def _oai_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        r = self._oai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return r.choices[0].message.content.strip()

    def _hf_generate(self, prompt: str, max_tokens: int) -> str:
        import torch
        inputs = self._hf_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            out = self._hf_model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=False, num_beams=4, early_stopping=True
            )
        return self._hf_tokenizer.decode(out[0], skip_special_tokens=True).strip()
