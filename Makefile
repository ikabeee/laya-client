.PHONY: setup start doctor dev install test lint format check docker-build docker-build-gpu up up-gpu down logs

setup:              ## detect the hardware, install torch + laya-client, download and test the model
	./scripts/setup.sh

start:              ## start the server with the settings in .env (fails fast if the model cannot run)
	.venv/bin/laya-client

doctor:             ## check torch, the GPU and the model without starting the server
	.venv/bin/laya-client doctor --load

dev:                ## auto-reloading server; every reload loads the model again
	.venv/bin/python -m uvicorn laya_client.interfaces.http.app:create_app --factory --reload --port 8000

install:            ## test and lint tooling only (no torch, no model): enough for `make check`
	uv venv --allow-existing .venv && uv pip install --python .venv -e ".[dev]"

test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

format:
	.venv/bin/ruff check --fix src tests
	.venv/bin/ruff format src tests

check: lint test

docker-build:       ## CPU image
	docker build -t laya-client .

docker-build-gpu:   ## NVIDIA image (CUDA 12.8 build of torch: RTX 50-series ready)
	docker build -t laya-client:gpu --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 .

up:                 ## run on the CPU with Docker Compose
	docker compose up -d --build

up-gpu:             ## run on an NVIDIA GPU with Docker Compose
	docker compose -f compose.yaml -f compose.gpu.yaml up -d --build

down:
	docker compose down

logs:
	docker compose logs -f laya-client
