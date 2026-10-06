# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Pinned, resolved dependency set (see requirements.lock) for reproducible builds.
COPY requirements.lock ./
RUN pip install -r requirements.lock

COPY pyproject.toml README.md alembic.ini ./
COPY alembic ./alembic
COPY config ./config
COPY app ./app
RUN pip install --no-deps .

RUN useradd --create-home --uid 10001 rag \
    && mkdir -p /models /data/blobs \
    && chown -R rag:rag /models /data /app
USER rag

EXPOSE 8000
CMD ["uvicorn", "app.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
