.PHONY: test lint fmt ci install

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff format .
	ruff check --fix .

test:
	pytest tests/ -v --cov --cov-report=term-missing

ci: lint test
