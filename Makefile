.PHONY: install install-dev test test-cov lint format train evaluate predict clean docker-build

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/brain_tumor_classifier --cov-report=term-missing --cov-report=html

lint:
	flake8 src/ tests/ scripts/
	mypy src/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

train:
	python scripts/train.py --config configs/config.yaml

evaluate:
	python scripts/evaluate.py --config configs/config.yaml --model-path artifacts/checkpoints/best_model.keras

predict:
	python scripts/predict.py --config configs/config.yaml --model-path artifacts/checkpoints/best_model.keras --image $(IMAGE)

docker-build:
	docker build -t brain-tumor-classifier:latest .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage build dist *.egg-info
