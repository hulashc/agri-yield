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

ENV PATH="/app/.venv/bin:"
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

# Use $PORT if set by Render, fallback to 8000 for local/Docker
CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
