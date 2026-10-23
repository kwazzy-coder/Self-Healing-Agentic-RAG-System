.PHONY: install dev test lint demo ui api eval bench-retrieval bench-detector ablations docker

install:            ## runtime dependencies
	pip install -r requirements.txt

dev:                ## lightweight test/lint dependencies (no torch)
	pip install -r requirements-dev.txt

test:
	pytest -q

lint:
	ruff check .

demo:
	python demo.py

ui:
	python ui.py

api:
	uvicorn api.server:app --reload --port 8000

eval:               ## self-healing vs vanilla on 100 HotpotQA questions
	python run_evaluation.py --dataset hotpotqa --n_samples 100 --compare_baseline

bench-retrieval:    ## BM25 vs dense vs hybrid (no API key)
	python -m evaluation.benchmark_retrieval --n_samples 500

bench-detector:     ## hallucination detection on RAGBench (heuristic+NLI needs no API key)
	python run_evaluation.py --dataset ragbench --subset covidqa --mode detector --n_samples 200 \
		--monitors heuristic,heuristic_nli,tiered,llm

ablations:
	python -m experiments.run_all_experiments

docker:
	docker build -t self-healing-rag .
