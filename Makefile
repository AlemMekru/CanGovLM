.PHONY: test lint

test:
	PYTHONPATH=src python3 -m unittest discover -s tests

lint:
	ruff check .
