# AI Codebase Intelligence Platform - Developer Tooling Makefile
.PHONY: help install test test-unit test-integration api ui eval docker-build docker-up docker-down clean lint

PYTHON ?= python3
PIP ?= pip
PYTEST ?= pytest

help:
	@echo "AI Codebase Intelligence Platform - Developer Tooling"
	@echo "====================================================="
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  make install            Install CPU PyTorch wheel and dependencies"
	@echo "  make test               Run the complete offline test suite"
	@echo "  make test-unit          Run unit tests excluding integration tests"
	@echo "  make test-integration   Run integration tests (requires network/API keys)"
	@echo "  make api                Launch FastAPI backend server (port 8000)"
	@echo "  make ui                 Launch Streamlit frontend UI (port 8501)"
	@echo "  make eval               Run benchmark evaluation on sample repository"
	@echo "  make docker-build       Build production Docker container image"
	@echo "  make docker-up          Launch container topology with docker-compose"
	@echo "  make docker-down        Stop docker-compose containers"
	@echo "  make clean              Remove temporary Python caches and build artifacts"
	@echo "  make lint               Run Python syntax checks across src/ and tests/"

install:
	$(PIP) install --upgrade pip
	$(PIP) install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) -v

test-unit:
	$(PYTEST) -v -m "not integration"

test-integration:
	$(PYTEST) -v -m "integration"

api:
	$(PYTHON) -m src.api.main --host 0.0.0.0 --port 8000

ui:
	$(PYTHON) -m src.ui.main --port 8501 --api-url http://localhost:8000

eval:
	$(PYTHON) -m src.evaluation.main --repo miguelgrinberg/microblog

docker-build:
	docker build -t ai-codebase-intel:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

lint:
	$(PYTHON) -m compileall src tests
