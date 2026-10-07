# syntax=docker/dockerfile:1.7
#
# Laya Client image.
#
#   docker build -t laya-client .                                  # CPU torch + real Laya engine (default)
#   docker build -t laya-client:gpu \
#       --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 .   # NVIDIA GPU
#   docker build -t laya-client:mock --build-arg ENGINE=mock .     # API + docs only, no torch (~150 MB)

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder
ARG ENGINE=laya
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# torch first, from the index that matches the target hardware, so `laya` does not pull the
# multi-gigabyte CUDA wheel onto a CPU-only VPS.
RUN if [ "$ENGINE" = "laya" ]; then \
        pip install --index-url "$TORCH_INDEX_URL" torch && pip install ".[engine]"; \
    else \
        pip install .; \
    fi

FROM python:${PYTHON_VERSION}-slim AS runtime
ARG ENGINE=laya
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/huggingface \
    USE_TF=0 \
    TOKENIZERS_PARALLELISM=false \
    LAYA_ENGINE=${ENGINE} \
    LAYA_HOST=0.0.0.0 \
    LAYA_PORT=8000
RUN useradd --create-home --uid 10001 laya \
    && mkdir -p /data/huggingface && chown -R laya:laya /data
COPY --from=builder /opt/venv /opt/venv
USER laya
WORKDIR /home/laya
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('LAYA_PORT','8000'), timeout=4)" || exit 1
CMD ["laya-client"]
