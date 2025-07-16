# Data Science Agent Makefile

.PHONY: help install install-dev setup clean test test-unit test-integration lint format type-check pre-commit run docs build

# Default target
help:
	@echo "Available commands:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  setup            Complete development setup"
	@echo "  clean            Clean up generated files"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run type checking with mypy"
	@echo "  pre-commit       Run pre-commit hooks"
	@echo "  run              Start the Streamlit application"
	@echo "  docs             Build documentation"
	@echo "  build            Build the package"

# Installation
install:
	pip install -r requirements.txt --no-deps

install-dev:
	pip install -e ".[dev,docs]"

setup: install-dev
	pre-commit install
	mkdir -p memory/vector memory/symbolic project_output/models project_output/reports project_output/data project_output/scripts
	@echo "Development environment setup complete!"

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

# Testing
test:
	pytest

test-unit:
	pytest tests/unit -m "not slow"

test-integration:
	pytest tests/integration

test-coverage:
	pytest --cov=agents --cov=memory --cov=ui --cov=utils --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 agents memory ui utils tests
	mypy agents memory ui utils

format:
	black agents memory ui utils tests main.py
	isort agents memory ui utils tests main.py

type-check:
	mypy agents memory ui utils

pre-commit:
	pre-commit run --all-files

# Application
run:
	streamlit run main.py

run-dev:
	streamlit run main.py --server.runOnSave true

# Documentation
docs:
	mkdocs build

docs-serve:
	mkdocs serve

# Building
build:
	python -m build

# Development helpers
check: lint test
	@echo "All checks passed!"

install-hooks:
	pre-commit install

init-project: setup
	@echo "Project initialized successfully!"
	@echo "Next steps:"
	@echo "1. Copy .env.example to .env and configure your API keys"
	@echo "2. Run 'make run' to start the application"