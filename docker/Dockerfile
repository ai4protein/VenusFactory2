# syntax=docker/dockerfile:1.7

# =============================================================================
# Stage 1 — build the React frontend (Node 25 to match frontend/package.json)
# =============================================================================
FROM node:25-slim AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build


# =============================================================================
# Stage 2 — runtime image
#
# Default base is CUDA 12.8 on Ubuntu 24.04 (ships python3.12).
# CPU-only build:
#   docker build \
#     --build-arg BASE_IMAGE=ubuntu:24.04 \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
#     --build-arg PYG_INDEX_URL=https://data.pyg.org/whl/torch-2.8.0+cpu.html \
#     -f docker/Dockerfile -t venusfactory/ce:cpu .
# =============================================================================
ARG BASE_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu24.04
FROM ${BASE_IMAGE} AS runtime

# Re-declare ARGs needed inside this stage
ARG TORCH_VERSION=2.8.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG PYG_INDEX_URL=https://data.pyg.org/whl/torch-2.8.0+cu128.html
ARG UV_VERSION=0.11.13
ARG TARGETARCH=amd64

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    WEBUI_V2_HOST=0.0.0.0 \
    WEBUI_V2_PORT=7861 \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin

# --- System packages -----------------------------------------------------
# python3.12 (Ubuntu 24.04 default), build tools, cairo/xml headers required by
# the pycairo build chain that xhtml2pdf pulls in.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential pkg-config \
        libcairo2-dev libxml2-dev libxslt1-dev libffi-dev libssl-dev \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- uv (standalone binary, no PEP 668 friction) --------------------------
RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) ARCH=x86_64;; \
        arm64) ARCH=aarch64;; \
        *) echo "unsupported arch: ${TARGETARCH}" && exit 1;; \
    esac; \
    curl -fsSL -o /tmp/uv.tar.gz \
        "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${ARCH}-unknown-linux-gnu.tar.gz"; \
    tar -xzf /tmp/uv.tar.gz -C /tmp; \
    mv "/tmp/uv-${ARCH}-unknown-linux-gnu/uv" "/tmp/uv-${ARCH}-unknown-linux-gnu/uvx" /usr/local/bin/; \
    rm -rf /tmp/uv.tar.gz "/tmp/uv-${ARCH}-unknown-linux-gnu"

# Project venv at a stable, non-/app path (so app code mounts don't clobber it)
RUN uv venv --python 3.12 "${VIRTUAL_ENV}"

WORKDIR /app

# --- PyTorch (separate layer for caching) --------------------------------
RUN uv pip install "torch==${TORCH_VERSION}" torchvision \
        --index-url "${TORCH_INDEX_URL}" --prerelease=allow

# --- PyG suite (separate layer) ------------------------------------------
RUN uv pip install \
        torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f "${PYG_INDEX_URL}"

# --- Project dependencies (cached unless pyproject.toml changes) ---------
COPY pyproject.toml ./
RUN uv pip install -r pyproject.toml

# --- Application code ----------------------------------------------------
COPY src/         ./src/
COPY scripts/     ./scripts/
COPY data/        ./data/
COPY install_config.json .env.example ./

# Built frontend assets from stage 1
COPY --from=frontend /build/dist/ ./frontend/dist/

# Provide a default .env so the app starts even when no .env is mounted.
# Mount a real .env via docker-compose env_file or `-v` to override.
RUN cp -n .env.example .env || true

EXPOSE 7861

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${WEBUI_V2_PORT}/health" || exit 1

CMD ["python", "-u", "src/webui_v2.py", "--host", "0.0.0.0", "--port", "7861"]
