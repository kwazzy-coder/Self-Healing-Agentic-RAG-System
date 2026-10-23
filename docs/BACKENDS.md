# LLM providers

All API providers are called through the **OpenAI-compatible Chat Completions API**
with the `openai` Python SDK, so switching is a configuration change:

```bash
cp .env.example .env      # then set LLM_BACKEND and one key
```

| `LLM_BACKEND` | Default model (`LLM_MODEL` overrides) | Cost | Key / setup |
|---|---|---|---|
| `groq` | `openai/gpt-oss-20b` | free tier | https://console.groq.com/keys |
| `gemini` | `gemini-3.5-flash-lite` | free tier | https://aistudio.google.com/apikey |
| `openai` | `gpt-5-mini` | paid | https://platform.openai.com/api-keys |
| `ollama` | `llama3.2` (`OLLAMA_MODEL`) | free, local | https://ollama.com — `ollama pull llama3.2` |
| `huggingface` | `google/flan-t5-base` | free, local | nothing (≈1 GB download, weak) |

If the selected provider has no key (or Ollama isn't running) the system falls
back in the order gemini → groq → openai → ollama → huggingface and logs a warning.

> Model availability changes. Google shut down `gemini-2.0-flash-lite` on 2026-06-01 and Groq
> retired `llama-3.1-8b-instant` on 2026-08-16 — the previous defaults of this project.
> Check [Gemini models](https://ai.google.dev/gemini-api/docs/models) /
> [Groq deprecations](https://console.groq.com/docs/deprecations) and set `LLM_MODEL` if a default disappears.

## Reasoning models

Gemini 3.x, `gpt-oss` and `gpt-5` models "think" before answering, and the thinking tokens count
toward the output budget. The client therefore

* sends `reasoning_effort` (default `low`, set `LLM_REASONING_EFFORT=` to omit it), and
* never asks for fewer than `MIN_OUTPUT_TOKENS` (1024) output tokens, so short prompts such as the
  YES/NO judge don't come back empty.

## Rate limits

Free tiers are rate-limited. The client waits between calls (`PROVIDER_MIN_DELAY` in `config.py`,
override with `MIN_DELAY_BETWEEN_CALLS`), retries 429/5xx responses with exponential back-off
(honouring `Retry-After`), and evaluations checkpoint every sample — if you hit a daily quota just
re-run the same command tomorrow and it resumes.

Rough budget: the full system makes ~1–2 LLM calls for an easy question and up to ~8 for a hard
multi-hop question that needs two corrections.

## Setting keys on Windows

```powershell
$env:GROQ_API_KEY="gsk_..."        # current PowerShell session
setx GROQ_API_KEY "gsk_..."        # permanently (open a new terminal afterwards)
```
or simply put it in `.env`, which `config.py` loads automatically.
