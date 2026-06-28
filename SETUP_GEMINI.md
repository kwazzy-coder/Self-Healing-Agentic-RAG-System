# Setting Up Gemini API (FREE — Best Quality)

## Step 1 — Get free API key (2 minutes, no credit card)
1. Go to: https://aistudio.google.com/apikey
2. Sign in with your Google/Gmail account
3. Click "Create API Key"
4. Click "Create API key in new project"
5. Copy the key — starts with "AIza..."

## Step 2 — Set the key

Windows PowerShell:
```
$env:GEMINI_API_KEY="AIzaxxxxxxxxxxxxxxxxxx"
```

Windows Command Prompt:
```
set GEMINI_API_KEY=AIzaxxxxxxxxxxxxxxxxxx
```

To make it permanent (recommended):
- Search "Environment Variables" in Windows Start menu
- Click "Edit the system environment variables"
- Click "Environment Variables"
- Under "User variables" click "New"
- Variable name: GEMINI_API_KEY
- Variable value: AIzaxxxxxxxxxxxxxxxxxx
- Click OK

## Step 3 — Install and run
```
pip install -r requirements.txt
python demo.py
python run_evaluation.py --dataset hotpotqa --n_samples 100 --compare_baseline
```

## Free Tier Limits
| Model | RPD | RPM | TPM |
|-------|-----|-----|-----|
| gemini-2.0-flash-lite | 1,000 | 15 | 250,000 |
| gemini-2.5-flash-lite | 1,000 | 15 | 250,000 |

100 samples × 5 calls = 500 calls → safely within 1,000 RPD
200 samples × 5 calls = 1,000 calls → uses full daily quota

## If you still have Groq key
Both keys can be set — system uses Gemini first, falls back to Groq if Gemini fails:
```
$env:GEMINI_API_KEY="AIza..."
$env:GROQ_API_KEY="gsk_..."
```

## Why Gemini is better for this project
- Better instruction following → shorter answers → higher EM score
- Better multi-hop reasoning → better on bridge questions
- 1 million token context window (vs 128K for Llama)
- Expected EM: 0.55-0.65 (vs 0.42 with Llama 8B)
