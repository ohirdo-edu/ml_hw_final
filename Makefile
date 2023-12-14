.PHONY: test
test:
	python -m pytest

.PHONY: coverage
coverage:
	python -m pytest --cov-report term --cov=src tests

.PHONY: lint
lint:
	ruff check src

.PHONY: online
online:
	python -m src.infer_online --model_url 'https://github.com/ohirdo-edu/ml_hw_final/raw/main/checkpoints/model_10001.joblib'
