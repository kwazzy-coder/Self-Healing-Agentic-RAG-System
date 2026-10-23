# Design decisions

Short answers to the "why did you do it this way?" questions.

### Why a *cascade* of checks instead of always asking an LLM judge?
An LLM judge costs one API call (seconds, money, rate limit) per answer. Most answers are easy to
classify: an empty answer or "I don't know" needs no model at all, and a small NLI model is
confident about clear-cut cases in ~50 ms on CPU. The judge is only needed for the uncertain band
of NLI scores. The `llm_judge_only` baseline and the `tier_distribution` in every report measure
how many calls this saves and whether accuracy changes.

### Why NLI and not embedding similarity for grounding?
Similarity says the answer and the passage are about the same thing; it does not say the passage
*supports* the answer ("Nolan is British" and "Nolan is American" are very similar). NLI is
trained for exactly the entailment question.

### Why turn a short answer into a claim before NLI?
"Paris" alone cannot be entailed by anything. "The answer to the question 'What is the capital of
France?' is Paris." can. v1 skipped NLI for answers of ≤ 2 words — almost every HotpotQA answer —
so Tier 2 effectively never ran.

### Why score every passage separately *and* a joint premise?
Concatenating all passages and cutting at 2,000 characters (v1) threw most of the context away and
exceeded what the model reads. Separate passages fit the model; the extra joint premise of the top
two passages lets two-hop evidence ("X directed Y" + "X was born in Canada") entail an answer
together.

### Why choose the strategy from the failure type?
A refusal means retrieval missed something — rephrasing the question helps. An unsupported answer
needs *different* evidence — a targeted query that excludes passages already seen. A wrong answer
type ("Christopher Nolan" for a yes/no question) usually means the question was misunderstood —
decomposition helps. Repeating the same strategy rarely helps, so strategies are never repeated.

### Why re-verify every correction and keep the best candidate?
A correction can make things worse. v1 returned the last correction without checking it. Now every
candidate is scored by the monitor and the best-grounded one is returned, so healing can only
replace an answer with one the monitor trusts more.

### Why `#1` substitution for bridge questions?
For "What nationality is the director of Inception?", hop 2 ("What nationality is #1?") cannot be
retrieved until hop 1 is answered. Substituting the hop-1 answer ("Christopher Nolan") gives the
retriever the entity it needs. This is the Self-Ask / decomposed-prompting idea.

### Why is the pipeline stateless?
v1 stored the document index on the retriever, so two concurrent API requests would overwrite each
other's documents. Now the index, corrector and cost tracker are created per call and the heavy
models are shared, so FastAPI can run requests in parallel threads. Per-request LLM usage is
tracked with a `ContextVar`, so costs never mix between threads (`test_concurrent_requests_*`).

### Why one OpenAI-compatible client for every provider?
Gemini, Groq, OpenAI and Ollama all expose the same Chat Completions API. One code path means one
retry/rate-limit implementation and switching provider is an env var. Both v1 default models were
shut down in 2026, which is exactly the situation where this pays off.

### Why paired significance tests?
Both systems answer the *same* questions, so their errors are correlated. McNemar's test (for EM)
and a paired bootstrap (for F1) use that pairing; with 100–200 questions a 2–3 point difference is
often not significant, and the reports say so.

### Why fakes instead of mocks in the tests?
`tests/fakes.py` provides a tiny hashing "embedding model", a lexical NLI model with the real label
layout, and an OpenAI-compatible fake client. The real `LLMReasoner` prompt-building and parsing
code runs in every test, the whole suite needs no network or GPU and finishes in seconds, and CI
installs no PyTorch. One test also starts a local HTTP server to exercise the real `openai` SDK
(request shape, 429 retry, auth errors).
