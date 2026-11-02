#!/usr/bin/env bash
# Resource partition/QOS/time are supplied by sbatch after live CCDS inspection.
#SBATCH --job-name=ai6130-ass2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/slurm-%j.log
#SBATCH --signal=B:USR1@180
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
# The model/method are passed as arguments: sbatch [verified flags] cluster/job.slurm tinyllama lora
model="${1:?}"; method="${2:?}"
pause_worker() { touch "runs/main-$model-$method-2ep/STOP_AFTER_STEP"; }
trap pause_worker USR1
bash cluster/run_cell.sh "$model" "$method" &
worker=$!
wait "$worker" || { result=$?; wait "$worker"; exit "$result"; }
