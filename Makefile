# ORION — Common Commands
# Usage: make <target>
# Windows: use 'nmake' or install GNU make

.PHONY: all install install-dev test test-unit test-integration test-smoke lint format typecheck clean preprocess train eval predict submit profile docs

# ─── Installation ─────────────────────────────────────────────────────────────
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,weak-supervision,deployment]"
	pre-commit install

# ─── Testing ──────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short -m "not slow"

test-integration:
	pytest tests/integration/ -v

test-smoke:
	pytest tests/smoke/ -v

test-cov:
	pytest --cov=src/orion --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

test-parallel:
	pytest tests/ -n auto --tb=short

# ─── Code Quality ─────────────────────────────────────────────────────────────
lint:
	ruff check src/ scripts/ tests/

format:
	black src/ scripts/ tests/ notebooks/
	ruff check --fix src/ scripts/ tests/

typecheck:
	mypy src/orion --ignore-missing-imports

pre-commit:
	pre-commit run --all-files

# ─── Data Pipeline ────────────────────────────────────────────────────────────
preprocess:
	python scripts/preprocess.py --config configs/data/preprocessing.yaml

extract-labels:
	python scripts/extract_weak_labels.py --config configs/data/preprocessing.yaml

eda:
	python scripts/eda.py --output-dir research/visualizations/eda

# ─── Training ─────────────────────────────────────────────────────────────────
train-baseline:
	python scripts/train.py --config configs/experiment/baseline_resnet50.yaml

train-swin:
	python scripts/train.py --config configs/experiment/swin_v2_base.yaml

train-multimodal:
	python scripts/train.py --config configs/experiment/multimodal_clip.yaml

train-cv:
	python scripts/cross_validate.py --config configs/experiment/swin_v2_base.yaml

# ─── Evaluation ───────────────────────────────────────────────────────────────
eval:
	python scripts/evaluate.py --checkpoint runs/best_model/checkpoint.pth

profile:
	python scripts/profile.py --config configs/experiment/swin_v2_base.yaml

# ─── Inference ────────────────────────────────────────────────────────────────
predict:
	python scripts/predict.py --checkpoint runs/best_model/checkpoint.pth --data-dir data/test

ensemble:
	python scripts/ensemble.py --config configs/inference/ensemble.yaml

submit:
	python scripts/submit.py --output submissions/submission_$(shell date +%Y%m%d_%H%M%S).csv

# ─── Export ───────────────────────────────────────────────────────────────────
export-onnx:
	python scripts/export.py --checkpoint runs/best_model/checkpoint.pth --format onnx

# ─── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker build -t orion:latest deployment/

docker-run:
	docker run --gpus all -p 8000:8000 orion:latest

# ─── Utilities ────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -rf dist/ build/

clean-runs:
	@echo "WARNING: This will delete all training runs!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -rf runs/

wandb-login:
	wandb login

wandb-sync:
	wandb sync --sync-all

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
