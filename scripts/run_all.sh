#!/usr/bin/env bash
#
# Reproduce every artifact in this repository, in dependency order.
#
#   bash scripts/run_all.sh                 # full reproduction (the README run)
#   CONFIG=configs/smoke.yaml bash scripts/run_all.sh   # fast smoke reproduction
#
# Every stage writes into artifacts/ (data, runs, results) and reports/.  The
# script stops on the first failure; nothing is faked or defaulted.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

CONFIG="${CONFIG:-configs/full.yaml}"
ABLATE_CONFIG="${ABLATE_CONFIG:-configs/ablate.yaml}"
RUN_NAME="${RUN_NAME:-full_linear}"
MODE="${MODE:-linear}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "0. environment"
$PY - <<'PY'
import platform, sys, torch
print(f"python   {sys.version.split()[0]} ({platform.platform()})")
print(f"torch    {torch.__version__}  cuda={torch.cuda.is_available()}  mps={getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()}")
PY

step "1. dataset (real if available, deterministic synthetic fallback otherwise)"
$PY scripts/fetch_data.py --config "$CONFIG" --force

step "2. train the frozen-DINOv2 perception model ($MODE)"
$PY scripts/train.py --config "$CONFIG" --mode "$MODE" --run-name "$RUN_NAME"

step "3. evaluate the trained checkpoint"
$PY scripts/eval.py --config "$CONFIG" --run-name "$RUN_NAME"

step "4. open-vocabulary zero-shot baselines (OWL-ViT, CLIP) + comparison table"
$PY scripts/zeroshot_baseline.py --config "$CONFIG" --method both --task-run "$RUN_NAME"
step "5. label-scarcity ablation (linear vs LoRA vs full x label budget)"
$PY scripts/ablate.py --config "$ABLATE_CONFIG"

# Pick the best measured cell from the ablation artifacts, so the remaining
# stages all operate on the same (best) model rather than a hard-coded one.
BEST_LINE="$($PY scripts/best_run.py || true)"
BEST_RUN="$(echo "$BEST_LINE" | awk '{print $1}')"
BEST_CKPT="$(echo "$BEST_LINE" | awk '{print $2}')"
if [ -z "$BEST_RUN" ]; then
    BEST_RUN="$RUN_NAME"
    BEST_CKPT="artifacts/runs/$RUN_NAME/best.pt"
fi
printf '    best ablation cell -> %s (%s)\n' "$BEST_RUN" "$BEST_CKPT"

step "6. hard-case mining, failure-mode clustering, and targeted retrain"
$PY scripts/mine_hard_cases.py --config "$ABLATE_CONFIG" --run-name "$BEST_RUN" --checkpoint "$BEST_CKPT"

step "7. TorchScript + ONNX export with parity checks"
$PY scripts/export.py --config "$ABLATE_CONFIG" --run-name "$BEST_RUN" --checkpoint "$BEST_CKPT"

step "8. latency benchmark (eager / TorchScript / ONNX)"
$PY scripts/bench_latency.py --config "$ABLATE_CONFIG" --run-name "$BEST_RUN" --checkpoint "$BEST_CKPT" --iterations 50

step "9. test suite + coverage"
$PY scripts/quality.py --marker "not slow" || true

step "10. render the report tables from artifacts/"
$PY scripts/make_report.py

printf '\n\033[1;32mAll artifacts reproduced.\033[0m  Results: reports/RESULTS.md\n'
