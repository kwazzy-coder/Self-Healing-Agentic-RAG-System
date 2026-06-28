# Setting Up Ollama (Free Local Llama 3.2)

## Step 1 — Install Ollama
Go to https://ollama.com/download and download for Windows.
Run the installer. Ollama runs as a background service automatically.

## Step 2 — Pull Llama 3.2
Open a new terminal and run:
```
ollama pull llama3.2
```
This downloads ~2GB. Wait for it to finish.

To verify it works:
```
ollama run llama3.2 "What is the capital of France?"
```
You should see: Paris

## Step 3 — Run the project
Open another terminal in the project folder:
```
python demo.py
python run_evaluation.py --dataset hotpotqa --n_samples 50 --compare_baseline
```

## Model options (pick based on your RAM)
| Model | RAM needed | Quality | Speed |
|-------|-----------|---------|-------|
| llama3.2:1b | 2GB | Good | Fast |
| llama3.2 (3b) | 4GB | Better | Medium |
| llama3.1:8b | 8GB | Best | Slow |

To use a different model, change in config.py:
```python
OLLAMA_MODEL = "llama3.2:1b"   # faster, less RAM
```

## Troubleshooting
- "Ollama not running" error → Open Ollama app from Start menu first
- "Model not found" → run: ollama pull llama3.2
- Slow responses → use llama3.2:1b instead
