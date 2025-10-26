# Makefile for Car Price Prediction project

.PHONY: all clean install test

all: install

install:
	pip install -r requirements.txt

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .venv

test:
	python -m pytest tests/

data:
	python scripts/prepare_data.py

train:
	python scripts/train_model.py

evaluate:
	python scripts/evaluate_model.py