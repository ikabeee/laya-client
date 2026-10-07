# syntax=docker/dockerfile:1.7
#
# Laya Client image. It always runs the real Laya model: at start-up the container checks torch and the
# GPU, downloads and loads the checkpoints and runs a test prediction, and exits with the reason if any
# of that fails.
#
#   docker build -t laya-client .                                   # CPU
#   docker build -t laya-client:gpu \
#       --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 .   # NVIDIA GPU (RTX 50xx ready)

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# torch first, from the index that matches the target hardware, so `laya` does not pull the
# multi-gigabyte default CUDA wheel onto a CPU-only VPS.
RUN pip install --index-url "$TORCH_INDEX_URL" torch && pip install ".[engine]"

FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/huggingface \
    USE_TF=0 \
    TOKENIZERS_PARALLELISM=false \
    LAYA_HOST=0.0.0.0 \
    LAYA_PORT=8000
RUN useradd --create-home --uid 10001 laya \
    && mkdir -p /data/huggingface && chown -R laya:laya /data
COPY --from=builder /opt/venv /opt/venv
USER laya
WORKDIR /home/laya
VOLUME ["/data"]
EXPOSE 8000
# The server only listens once the model is loaded and tested; the first start downloads the weights.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/ready' % os.environ.get('LAYA_PORT','8000'), timeout=4)" || exit 1
CMD ["laya-client"]
