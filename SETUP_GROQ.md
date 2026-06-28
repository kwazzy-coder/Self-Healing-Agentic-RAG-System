# Setting Up Groq API (FREE — Llama 3.3 70B)

## Step 1 — Get your free API key (2 minutes)
1. Go to https://console.groq.com
2. Sign up with Google/GitHub — no credit card needed
3. Click "API Keys" → "Create API Key"
4. Copy the key (starts with gsk_...)

## Step 2 — Set the key in your terminal

Windows (Command Prompt):
```
set GROQ_API_KEY=gsk_your_key_here
```

Windows (PowerShell):
```
$env:GROQ_API_KEY="gsk_your_key_here"
```

To make it permanent (so you don't have to set it every time):
- Search "Environment Variables" in Windows
- Add new variable: GROQ_API_KEY = gsk_your_key_here

## Step 3 — Install and run
```
pip install -r requirements.txt
python demo.py
python run_evaluation.py --dataset hotpotqa --n_samples 100 --compare_baseline
```

## Free tier limits (more than enough for this project)
| Model | Requests/min | Requests/day | Tokens/min |
|-------|-------------|-------------|-----------|
| llama-3.3-70b-versatile | 30 | 14,400 | 6,000 |
| llama-3.1-8b-instant | 30 | 14,400 | 30,000 |

100 samples × ~4 calls each = 400 calls → well within 14,400/day limit.
The code already adds a 2.2s delay between calls to stay under 30 RPM.

## If you hit rate limits
Change model in config.py to the faster 8B model:
```python
GROQ_MODEL = "llama-3.1-8b-instant"
```
This has 5x higher token limits but slightly lower quality.

## No Groq account? Use Ollama instead
See SETUP_OLLAMA.md — runs locally, zero rate limits, needs 4GB RAM.
