#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
model="${1:?model: tinyllama/openllama}"
method="${2:?method: lora/adapterp}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python_bin="${ASS2_PYTHON:-$PWD/.venv-ass2/bin/python}"
mkdir -p logs
"$python_bin" tools/train.py --model "$model" --method "$method" --resume 2>&1 | tee -a "logs/$model-$method-training.log"
run="runs/main-$model-$method-2ep"
test -f "$run/training-complete.json"
for task in AQuA AddSub; do
  "$python_bin" tools/evaluate.py --run "$run" --task "$task" 2>&1 | tee -a "logs/$model-$method-$task.log"
done
"$python_bin" tools/summarize.py
