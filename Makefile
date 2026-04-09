.PHONY: setup lint test test-cov pre-commit clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -r requirements-dev.txt
	.venv/bin/python -c "import nltk; nltk.download('punkt')"
	.venv/bin/pre-commit install

lint:
	.venv/bin/python -m flake8 src/ tests/ --max-line-length 100
	.venv/bin/python -m mypy src/ --ignore-missing-imports

test:
	.venv/bin/python -m pytest tests/ -v --tb=short

test-cov:
	.venv/bin/python -m pytest tests/ -v --tb=short --cov=words_of_war --cov-report=term-missing

pre-commit:
	.venv/bin/pre-commit run --all-files

clean:
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*.pyc' -delete
	find . -name '.ipynb_checkpoints' -exec rm -rf {} +
