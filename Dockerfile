# syntax=docker/dockerfile:1

FROM python:3.14-slim-bookworm AS builder

# Grab the uv binary from Astral's official image instead of pip-installing it.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first, separately from the app code, so this layer
# only gets invalidated when pyproject.toml/uv.lock actually change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Now add the source and install the project itself.
# README.md is required too: pyproject.toml declares it as the package readme,
# and the build backend refuses to build without it.
COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev


FROM python:3.14-slim-bookworm

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH"

USER appuser
EXPOSE 8000

CMD ["uvicorn", "llm_service_fastapi.api:app", "--host", "0.0.0.0", "--port", "8000"]
