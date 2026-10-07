.PHONY: install install-engine dev run test lint format check docker-build docker-build-mock up down logs

PY ?= .venv/bin/python

install:            ## API, docs and test tooling (mock engine only)
	uv venv .venv && uv pip install --python .venv -e ".[dev]"

install-engine:     ## add the real Laya engine with CPU torch
	uv pip install --python .venv --index-url https://download.pytorch.org/whl/cpu torch
	uv pip install --python .venv -e ".[engine,dev]"

dev:                ## hot-reload server on the mock engine
	LAYA_ENGINE=mock $(PY) -m uvicorn laya_client.interfaces.http.app:create_app --factory --reload --port 8000

run:                ## server with the settings from .env
	$(PY) -m laya_client

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format:
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

check: lint test

docker-build:
	docker build -t laya-client .

docker-build-mock:
	docker build -t laya-client:mock --build-arg ENGINE=mock .

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f laya-client
