"""
agent/llm_reasoner.py
KEY CHANGE: Chain-of-Thought (CoT) prompting.
Forces LLM to reason step-by-step before giving final answer.
This is the single biggest improvement for multi-hop QA.

Expected improvement: +15-20% EM, +10-15% F1
"""
import os, time, threading, re
from config import (
    LLM_BACKEND,
    GEMINI_API_KEY, GEMINI_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
    OLLAMA_MODEL, OLLAMA_BASE_URL,
    OPENAI_MODEL, OPENAI_API_KEY, HF_MODEL,
    MIN_DELAY_BETWEEN_CALLS, MAX_RETRIES, RETRY_WAIT_SECONDS,
)

# Chain-of-Thought few-shot examples
# Show the model HOW to reason before answering
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
    def __init__(self, min_delay):
        self._lock = threading.Lock()
        self._last = 0.0
        self.min_delay = min_delay

    def wait(self):
        with self._lock:
            gap = self.min_delay - (time.time() - self._last)
            if gap > 0 and self._last > 0:
                time.sleep(gap)
            self._last = time.time()


_limiter = _RateLimiter(MIN_DELAY_BETWEEN_CALLS)


class LLMReasoner:
    def __init__(self, backend=None):
        self.backend        = backend or LLM_BACKEND
        self._gemini_client = None
        self._groq_client   = None
        self._oai_client    = None
        self._hf_tokenizer  = None
        self._hf_model      = None
        self._init()

    def _init(self):
        if self.backend == "gemini":
            if self._init_gemini(): print(f"[LLMReasoner] ✓ Gemini: {GEMINI_MODEL}")
            else: self.backend = "groq"; self._init()
        elif self.backend == "groq":
            if self._init_groq(): print(f"[LLMReasoner] ✓ Groq: {GROQ_MODEL}")
            else: self.backend = "ollama"; self._init()
        elif self.backend == "ollama":
            if self._check_ollama(): print(f"[LLMReasoner] ✓ Ollama: {OLLAMA_MODEL}")
            else: self.backend = "openai"; self._init()
        elif self.backend == "openai":
            if self._init_openai(): print(f"[LLMReasoner] ✓ OpenAI: {OPENAI_MODEL}")
            else: self.backend = "huggingface"; self._init_hf()
        else:
            self._init_hf()

    def _init_gemini(self):
        key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        if not key:
            print("[LLMReasoner] No GEMINI_API_KEY → https://aistudio.google.com/apikey")
            return False
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            self._gemini_client = genai.GenerativeModel(GEMINI_MODEL)
            self._gemini_client.generate_content(
                "hi", generation_config={"max_output_tokens": 3})
            return True
        except ImportError:
            print("[LLMReasoner] Run: pip install google-generativeai"); return False
        except Exception as e:
            print(f"[LLMReasoner] Gemini error: {e}"); return False

    def _init_groq(self):
        key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        if not key:
            print("[LLMReasoner] No GROQ_API_KEY → https://console.groq.com")
            return False
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=key)
            self._groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role":"user","content":"hi"}], max_tokens=3)
            return True
        except ImportError:
            print("[LLMReasoner] Run: pip install groq"); return False
        except Exception as e:
            print(f"[LLMReasoner] Groq error: {e}"); return False

    def _check_ollama(self):
        try:
            import requests
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return any(OLLAMA_MODEL.split(":")[0] in m for m in models)
        except Exception: pass
        return False

    def _init_openai(self):
        key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        if not key: return False
        try:
            from openai import OpenAI
            self._oai_client = OpenAI(api_key=key); return True
        except ImportError: return False

    def _init_hf(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        print(f"[LLMReasoner] Loading {HF_MODEL}...")
        self._hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        self._hf_model     = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL)
        print("[LLMReasoner] ✓ HuggingFace loaded.")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, prompt, max_tokens=128, temperature=0.0):
        _limiter.wait()
        for attempt in range(MAX_RETRIES):
            try:
                if   self.backend == "gemini": return self._gemini_generate(prompt, max_tokens, temperature)
                elif self.backend == "groq":   return self._groq_generate(prompt, max_tokens, temperature)
                elif self.backend == "ollama": return self._ollama_generate(prompt, max_tokens, temperature)
                elif self.backend == "openai": return self._oai_generate(prompt, max_tokens, temperature)
                else:                          return self._hf_generate(prompt, max_tokens)
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "rate" in msg or "quota" in msg or "too many" in msg:
                    wait = self._parse_retry_after(e) or (RETRY_WAIT_SECONDS * (attempt+1))
                    print(f"[LLMReasoner] Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[LLMReasoner] Error: {e}")
                    return "I don't know."
        return "I don't know."

    def _parse_retry_after(self, error):
        try:
            m = re.search(r"(\d+\.?\d*)\s*s", str(error), re.IGNORECASE)
            return int(float(m.group(1))) + 2 if m else None
        except Exception: return None

    def answer_with_context(self, question, context_passages):
        """
        Chain-of-Thought answering:
        1. Show few-shot CoT examples
        2. Give context + question
        3. Ask model to reason step-by-step
        4. Extract final answer from reasoning
        """
        limit    = 3000 if self.backend in ("gemini","groq","openai","ollama") else 600
        combined = " ".join(p.strip() for p in context_passages)[:limit]

        if self.backend == "huggingface":
            raw = self.generate(
                f"Context: {combined}\n\nQuestion: {question}\nAnswer:",
                max_tokens=32)
            return self._extract_final_answer(raw)

        # Chain-of-Thought prompt
        prompt = (
            f"{COT_FEW_SHOT}"
            f"Now answer this question using the context below.\n"
            f"First write your Reasoning (1-2 sentences), "
            f"then write Answer (1-4 words only).\n\n"
            f"Context: {combined}\n\n"
            f"Q: {question}\n"
            f"Reasoning:"
        )
        # Allow more tokens for reasoning
        raw = self.generate(prompt, max_tokens=80)
        return self._extract_final_answer(raw)

    def answer_with_sub_answers(self, question, sub_answers, context_passages):
        limit    = 2500 if self.backend in ("gemini","groq","openai","ollama") else 400
        combined = " ".join(p.strip() for p in context_passages)[:limit]
        hints    = " | ".join(sub_answers)

        if self.backend == "huggingface":
            raw = self.generate(
                f"Context: {combined} {hints}\n\nQuestion: {question}\nAnswer:",
                max_tokens=32)
            return self._extract_final_answer(raw)

        prompt = (
            f"{COT_FEW_SHOT}"
            f"Use the context and intermediate findings.\n"
            f"First write Reasoning (1-2 sentences), then Answer (1-4 words).\n\n"
            f"Context: {combined}\n"
            f"Findings: {hints}\n\n"
            f"Q: {question}\n"
            f"Reasoning:"
        )
        raw = self.generate(prompt, max_tokens=80)
        return self._extract_final_answer(raw)

    def _extract_final_answer(self, raw: str) -> str:
        """
        Extract the Answer from CoT output.
        CoT output looks like:
          "The film was directed by X. X was born in Y.
           Answer: Y"
        We extract whatever comes after "Answer:"
        """
        if not raw:
            return ""

        # Try to find explicit "Answer:" marker
        if "answer:" in raw.lower():
            parts = re.split(r"answer\s*:", raw, flags=re.IGNORECASE)
            if len(parts) > 1:
                answer_part = parts[-1].strip()
                # Take just first line of the answer
                answer_part = answer_part.split("\n")[0].strip()
                return self._clean(answer_part)

        # No "Answer:" found — treat whole thing as answer
        # Take last sentence/phrase (likely the conclusion)
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        if lines:
            return self._clean(lines[-1])

        return self._clean(raw)

    def _clean(self, answer: str) -> str:
        """Clean up the extracted answer."""
        if not answer:
            return ""
        a = answer.strip()

        # Remove quotes
        a = a.strip('"\'')

        # Remove common prefixes
        prefixes = [
            r"^the answer is[:\s]+",
            r"^answer[:\s]+",
            r"^so the answer is[:\s]+",
            r"^therefore[,\s]+",
            r"^thus[,\s]+",
        ]
        for pat in prefixes:
            a = re.sub(pat, "", a, flags=re.IGNORECASE).strip()

        # yes/no: keep just the word
        if re.match(r"^(yes|no)[,\.\s]", a, re.IGNORECASE):
            return a.split()[0].lower()

        # "X is the Y of Z" → "X"
        match = re.match(
            r"^(.+?)\s+(is|was|are|were)\s+(the|a|an)\s+",
            a, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if 1 <= len(candidate.split()) <= 5:
                a = candidate

        # Remove parenthetical info
        a = re.sub(r"\s*\(.*?\)", "", a)

        # Remove trailing punctuation
        a = a.strip().rstrip(".,;:\"'")

        # If still too long take first 5 words
        words = a.split()
        if len(words) > 6:
            a = " ".join(words[:5])

        return a.strip()

    # ── Backends ──────────────────────────────────────────────────────────────

    def _gemini_generate(self, prompt, max_tokens, temperature):
        import google.generativeai as genai
        response = self._gemini_client.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        )
        return response.text.strip()

    def _groq_generate(self, prompt, max_tokens, temperature):
        r = self._groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens, temperature=temperature,
        )
        return r.choices[0].message.content.strip()

    def _ollama_generate(self, prompt, max_tokens, temperature):
        import requests
        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate",
            json={"model":OLLAMA_MODEL,"prompt":prompt,"stream":False,
                  "options":{"num_predict":max_tokens,"temperature":temperature}},
            timeout=60)
        r.raise_for_status()
        return r.json().get("response","").strip()

    def _oai_generate(self, prompt, max_tokens, temperature):
        r = self._oai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens, temperature=temperature,
        )
        return r.choices[0].message.content.strip()

    def _hf_generate(self, prompt, max_tokens):
        import torch
        inputs = self._hf_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = self._hf_model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=False, num_beams=4, early_stopping=True)
        return self._hf_tokenizer.decode(out[0], skip_special_tokens=True).strip()
