#!/usr/bin/env bash
# One-command setup for running laya-client on this machine (Linux, WSL2 or macOS).
#
#   ./scripts/setup.sh            # detect the hardware, install, download and test the model
#   ./scripts/setup.sh --dry-run  # only print what would be installed
#
# 1. Picks the torch build for this machine: CUDA (cu128 by default, required by RTX 50-series GPUs)
#    when an NVIDIA GPU is present, Apple's default build on macOS, the CPU build otherwise.
# 2. Creates .venv and installs torch plus laya-client with the Laya engine.
# 3. Writes .env from .env.example (if missing) with the detected LAYA_DEVICE.
# 4. Runs `laya-client doctor --load`: downloads the checkpoints and runs a test prediction on the
#    chosen device. Any problem stops the script with a non-zero exit code and the reason.
#
# Overrides (environment variables):
#   LAYA_DEVICE=auto|cuda|cpu|mps   hardware to install for (default: auto-detect)
#   TORCH_CUDA=cu128                CUDA build of torch for NVIDIA GPUs (cu128, cu129, cu130 ...)
#   PYTHON=python3                  interpreter used when uv is not installed
#   LAYA_MODELS=english,multilingual   checkpoints to download and test

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

DEVICE="${LAYA_DEVICE:-auto}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
PYTHON="${PYTHON:-python3}"
VENV=".venv"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
run() {
  printf '    $ %s\n' "$*"
  if [[ "$DRY_RUN" == 0 ]]; then "$@"; fi
}

# --- 1. Hardware ----------------------------------------------------------------------------------
has_nvidia_gpu() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }

if [[ "$DEVICE" == "auto" ]]; then
  if has_nvidia_gpu; then
    DEVICE="cuda"
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    DEVICE="auto"   # macOS: the default PyPI build uses MPS on Apple silicon, CPU on Intel
  else
    DEVICE="cpu"
  fi
fi

case "$DEVICE" in
  cuda*)
    has_nvidia_gpu || die "LAYA_DEVICE=$DEVICE but nvidia-smi finds no NVIDIA GPU. Install the NVIDIA driver (on WSL2, the Windows driver) and retry."
    log "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader | sed 's/^/    /'
    TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"
    ;;
  cpu)
    if [[ "$(uname -s)" == "Darwin" ]]; then TORCH_INDEX=""; else TORCH_INDEX="https://download.pytorch.org/whl/cpu"; fi
    has_nvidia_gpu && warn "an NVIDIA GPU is present but LAYA_DEVICE=cpu: installing the CPU build on purpose."
    log "No GPU selected: installing the CPU build of torch."
    ;;
  auto | mps)
    [[ "$(uname -s)" == "Darwin" ]] || die "LAYA_DEVICE=$DEVICE is only supported on macOS."
    TORCH_INDEX=""
    log "macOS: installing the default torch build (MPS on Apple silicon)."
    ;;
  *)
    die "unknown LAYA_DEVICE=$DEVICE: use auto, cuda, cpu or mps"
    ;;
esac

# --- 2. Python environment ------------------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
  log "Creating $VENV with uv"
  run uv venv --allow-existing --python ">=3.10" "$VENV"
  PIP=(uv pip install --python "$VENV")
else
  command -v "$PYTHON" >/dev/null 2>&1 || die "$PYTHON not found: install Python 3.10+ or uv (https://docs.astral.sh/uv/)"
  "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
    || die "Python 3.10+ is required (found $("$PYTHON" -V 2>&1))"
  log "Creating $VENV with $PYTHON -m venv"
  run "$PYTHON" -m venv "$VENV"
  run "$VENV/bin/python" -m pip install --upgrade pip
  PIP=("$VENV/bin/python" -m pip install)
fi

log "Installing torch${TORCH_INDEX:+ from $TORCH_INDEX}"
if [[ -n "$TORCH_INDEX" ]]; then
  run "${PIP[@]}" --index-url "$TORCH_INDEX" torch
else
  run "${PIP[@]}" torch
fi

log "Installing laya-client with the Laya engine"
run "${PIP[@]}" -e ".[engine,dev]"

# --- 3. Configuration -----------------------------------------------------------------------------
if [[ ! -f .env ]]; then
  log "Writing .env from .env.example (LAYA_DEVICE=$DEVICE)"
  if [[ "$DRY_RUN" == 0 ]]; then
    sed "s/^LAYA_DEVICE=.*/LAYA_DEVICE=${DEVICE}/" .env.example > .env
  fi
else
  log ".env already exists: leaving it unchanged (detected device: $DEVICE)"
fi

# --- 4. Verify: download the checkpoints and run a prediction on the chosen device --------------
log "Checking the runtime and loading the model (the first run downloads 1-2 GB of weights)"
if [[ "$DRY_RUN" == 0 ]]; then
  LAYA_DEVICE="$DEVICE" "$VENV/bin/laya-client" doctor --load \
    || die "the model could not run on this machine; see the message above."
  log "Done. Start the server with: make start   (API reference at http://127.0.0.1:8000/docs)"
else
  printf '    $ LAYA_DEVICE=%s %s/bin/laya-client doctor --load\n' "$DEVICE" "$VENV"
fi
