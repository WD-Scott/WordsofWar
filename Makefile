.PHONY: setup lint clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python -c "import nltk; nltk.download('punkt')"

lint:
	.venv/bin/pip install --quiet flake8 mypy
	.venv/bin/python -m flake8 Python_Modules/ --max-line-length 100
	.venv/bin/python -m mypy Python_Modules/

clean:
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*.pyc' -delete
	find . -name '.ipynb_checkpoints' -exec rm -rf {} +
