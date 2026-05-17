FROM python:3.13-slim AS builder
WORKDIR /app

RUN pip install uv
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --python /usr/local/bin/python3.13

FROM python:3.13-slim AS runtime
WORKDIR /app

COPY --from=builder /app/.venv ./.venv

# Copy all application modules needed at runtime
COPY serving/ ./serving/
COPY ingestion/ ./ingestion/
COPY monitoring/ ./monitoring/
COPY training/ ./training/
COPY features/feast_repo/ ./features/feast_repo/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

# Use $PORT if set by Render, fallback to 8000 for local/Docker
CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
