# AerialObstacleViT -- convenience wrappers around the same commands as scripts/run_all.sh.
#
#   make install      create the venv and install the package
#   make all          reproduce every artifact (data -> report)
#   make test         unit tests only (no downloads)
#   make cov          unit tests with coverage
#   make docker-build / make docker-run

PY      ?= .venv/bin/python
CONFIG  ?= configs/full.yaml
ABLATE  ?= configs/ablate.yaml
SMOKE   ?= configs/smoke.yaml
RUN     ?= full_linear

.DEFAULT_GOAL := help
.PHONY: help install data data-smoke train eval zeroshot ablate hardcases export bench \
        quality test cov lint fmt lint-fix report all docker-build docker-run clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## create .venv (py3.11) and install the package + dev extras
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"

data: ## fetch the real dataset or generate the synthetic fallback
	$(PY) scripts/fetch_data.py --config $(CONFIG) --force

data-smoke: ## tiny dataset for CI / quick checks
	$(PY) scripts/fetch_data.py --config $(SMOKE) --force

train: ## train the perception model (MODE=linear|lora_rN|full)
	$(PY) scripts/train.py --config $(CONFIG) --mode $(MODE) --run-name $(RUN)

eval: ## evaluate a trained checkpoint on the test split
	$(PY) scripts/eval.py --config $(CONFIG) --run-name $(RUN)

zeroshot: ## run the OWL-ViT and CLIP zero-shot baselines
	$(PY) scripts/zeroshot_baseline.py --config $(CONFIG) --method both --task-run $(RUN)

ablate: ## label-scarcity ablation sweep
	$(PY) scripts/ablate.py --config $(ABLATE)

hardcases: ## mine, cluster and tag hard cases, then retrain
	$(PY) scripts/mine_hard_cases.py --config $(ABLATE) --run-name $(RUN)

export: ## TorchScript + ONNX export with parity checks
	$(PY) scripts/export.py --config $(CONFIG) --run-name $(RUN)

bench: ## latency benchmark
	$(PY) scripts/bench_latency.py --config $(CONFIG) --run-name $(RUN) --iterations 50

quality: ## tests + coverage, recorded to artifacts/results/quality.json
	$(PY) scripts/quality.py

test: ## unit tests (slow/download tests deselected)
	$(PY) -m pytest -q -m "not slow"

cov: ## unit tests with coverage
	$(PY) -m pytest -q -m "not slow" --cov=aerial_vit --cov-report=term-missing

lint: ## ruff check
	$(PY) -m ruff check .

fmt: ## ruff format
	$(PY) -m ruff format .

lint-fix: ## ruff check --fix + format
	$(PY) -m ruff check --fix . && $(PY) -m ruff format .

report: ## render reports/RESULTS.md and reports/tables/*.md from artifacts
	$(PY) scripts/make_report.py

all: ## reproduce every artifact in order
	bash scripts/run_all.sh

docker-build: ## build the CPU container image
	docker build -t aerialvit .

docker-run: ## verify the image entrypoint
	docker run --rm aerialvit --help

clean: ## remove generated data, runs and caches (keeps results JSON/CSV)
	rm -rf artifacts/data artifacts/runs artifacts/tmp artifacts/models/*.onnx \
	       artifacts/models/*.torchscript .pytest_cache .ruff_cache coverage.json
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
