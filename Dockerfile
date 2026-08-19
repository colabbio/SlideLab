# SlideLab — WSI Preprocessing Pipeline
#
# Build targets:
#   CPU-only (smaller, for dev/CI):
#     docker build --target cpu -t slidelab:cpu .
#
#   GPU (CUDA 11.8, for production):
#     docker build --target gpu -t slidelab:gpu .
#
# Usage:
#   docker run --rm -v /data/slides:/input -v /data/output:/output \
#     slidelab:cpu \
#     -i /input -o /output -s 256 -m 20 \
#     --normalize_staining macenko --device cpu
#
#   # GPU with NVIDIA runtime:
#   docker run --rm --gpus all \
#     -v /data/slides:/input -v /data/output:/output \
#     slidelab:gpu \
#     -i /input -o /output -s 256 -m 20 \
#     --normalize_staining macenko --device cuda

# ─── Stage 1: Base with system dependencies ──────────────────────────────────
FROM python:3.10-slim AS base

LABEL maintainer="Lorenzo Olmo Marchal <lolmomarchal@ucsd.edu>"
LABEL description="SlideLab: Whole Slide Image preprocessing pipeline"
LABEL org.opencontainers.image.source="https://github.com/lolmomarchal/SlideLab"

# System deps for OpenSlide, OpenCV, and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libopenslide-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libffi-dev \
    libvips-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Stage 2: CPU-only variant ────────────────────────────────────────────────
FROM base AS cpu

# Install CPU-only PyTorch first (keeps image ~1.5GB vs ~6GB with CUDA)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY SlideLab/ /app/SlideLab/
COPY __init__.py /app/

ENV PYTHONPATH="/app:/app/SlideLab"

WORKDIR /app/SlideLab

# Health check: verify OpenSlide is importable
RUN python -c "import openslide; print(f'OpenSlide {openslide.__library_version__}')"

CMD ["python", "SlidePreprocessing.py", "--help"]

# ─── Stage 3: GPU variant (CUDA 11.8) ────────────────────────────────────────
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS gpu-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libopenslide-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libvips-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

FROM gpu-base AS gpu

# Install CUDA-enabled PyTorch
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install remaining Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY SlideLab/ /app/SlideLab/
COPY __init__.py /app/

WORKDIR /app/SlideLab

# Health check
RUN python -c "import openslide; import torch; print(f'OpenSlide OK, CUDA available: {torch.cuda.is_available()}')"

ENTRYPOINT ["python", "SlidePreprocessing.py"]
CMD ["--help"]
