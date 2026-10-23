# Self-Healing Agentic RAG — API image (CPU)
#   docker build -t self-healing-rag .
#   docker run -p 8000:8000 --env-file .env self-healing-rag
# Add --build-arg PRELOAD_MODELS=1 to bake the embedding + NLI models into the image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

ARG PRELOAD_MODELS=0
RUN if [ "$PRELOAD_MODELS" = "1" ]; then python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/nli-deberta-v3-small')"; fi

COPY . .
RUN useradd --create-home appuser && mkdir -p /models && chown -R appuser /app /models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
