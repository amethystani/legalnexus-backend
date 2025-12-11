.PHONY: help install dev test lint clean run eval docker

help:
	@echo "LegalNexus - Available commands:"
	@echo ""
	@echo "  make install    Install production dependencies"
	@echo "  make dev        Install development dependencies"
	@echo "  make test       Run tests"
	@echo "  make lint       Run linters"
	@echo "  make eval       Run evaluation script"
	@echo "  make run        Run the main application"
	@echo "  make clean      Remove cache files"
	@echo "  make docker     Build Docker image"
	@echo ""

install:
	pip install -r config/requirements.txt

dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	python -m pytest tests/ -v

lint:
	flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
	black --check src/
	isort --check-only src/

format:
	black src/
	isort src/

eval:
	python src/evaluation/real_evaluation.py

run:
	python src/ui/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

docker:
	docker build -t legalnexus:latest .
