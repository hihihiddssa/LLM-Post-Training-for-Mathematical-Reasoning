# AI6130 Assignment 2 - completed experiments

Student: AGA CILA, G2605698H. Completed 1 November 2026; evidence verified 2 November 2026.

## Open these first

- `report/AI6130-Ass2-G2605698H.pdf`: final English report, cover plus four body pages.
- `report/AI6130-Ass2-G2605698H.md`: editable report source.
- `results/final/summary.md`, `summary.csv`, `summary.json`: six configurations, original and audited scores, actual parameters, loss, time and memory.
- `collected/20261102/runs/`: four final adapters, all original raw predictions, metadata and training histories; both baselines.
- `results/final/audited-predictions/`: separate post-generation scoring audit. Original files are immutable.
- `results/final/parser-changes.json`: every changed prediction, including incorrect conclusions.
- `collected/experiment-delivery-20261102.tar.gz`: full downloaded evidence archive, excluding intermediate checkpoints.

All four runs are preparation complete, training complete, evaluation complete and results summarized. Baselines required no training and have both evaluations complete. All 3,894 records were checked for coverage, unique indices, unchanged gold answers, question hashes, original parser outputs and correctness. Four final weight SHA256 hashes match training metadata. Intermediate resumable optimizer checkpoints remain on NSCC.

## Reproduce on NSCC

Teacher source is the sibling `AI6130_Assignment2/`. Source commit: 21086af375ddb6433e8dd00df0a28eeb5ce65d25. The historical initial deployment archive plus downloaded `tools/` snapshot preserve the executed source. Do not overwrite completed runs: an identity mismatch is intentional protection. For independent reruns use a new release directory, keeping the same sibling layout.

From GPU/, activate the dedicated environment (see requirements-compatible.txt and cluster/prepare.pbs). Actual tested environment: Python 3.12, torch 2.8.0+cu126, transformers 4.38.0, accelerate 0.28.0, datasets 2.18.0, teacher PEFT fork. Actual GPU: A100-SXM4-40GB. Model revisions, dataset hashes and fixed split indices are in artifacts/. No external commercial API is used.

```bash
export HF_HOME="$HOME/AI6130-Ass2-G2605698H/hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="$HOME/AI6130-Ass2-G2605698H/env/bin/python"
$PY tools/audit.py
$PY tools/test_protocol.py
$PY tools/smoke.py
# In a fresh release, repeat for tinyllama/openllama and lora/adapterp:
$PY tools/train.py --model tinyllama --method lora --smoke
$PY tools/train.py --model tinyllama --method lora --resume
$PY tools/evaluate.py --run runs/main-tinyllama-lora-2ep --task AQuA
$PY tools/evaluate.py --run runs/main-tinyllama-lora-2ep --task AddSub
# Repeat baseline for each model and each task; this never injects an adapter:
$PY tools/evaluate_baseline.py --model tinyllama --task AQuA
$PY tools/evaluate_baseline.py --model tinyllama --task AddSub
```

PBS scripts request one GPU, 16 CPUs and 110 GB memory using personal-cila001. Existing submitted jobs must not be resubmitted to poll status. cluster/main.pbs runs training followed by both benchmarks; cluster/baseline.pbs runs both baseline benchmarks. All shell files must use LF line endings. A dedicated two-step interruption/resume check passed before formal training. CUDA kernel caches are isolated/bypassed in runtime.py following the tested A100 reduction fix.

## Reproduce the evidence audit and report locally

Use Python 3.12 for the stdlib-only audit (training libraries are unnecessary):

```powershell
python tools/scoring_audit.py
python tools/finalize_results.py
python tools/build_report.py
```

finalize_results.py reads the downloaded `collected/20261102/` evidence and unchanged sibling datasets. build_report.py uses reportlab, matplotlib and pymupdf from `report/python-deps/` (or an independent report environment). Versions actually used are recorded in report/report-environment.json. It produces the PDF, Markdown source, validation-loss figure and page previews.

Scoring v1 is unchanged. Conclusion audit v2 adds one-number, complete final conclusion lines for AddSub only; rejects equations, multiple numbers, unfinished thoughts and conflicting explicit numerical answers. It never selects an arbitrary last reasoning number and never consults gold to select a prediction. Nine synthetic examples and a fixed-seed sample of 20 changed outputs were inspected. All 1,005 changes, including wrong answers, are retained. This is a post-generation extraction audit, not an additional training or decoding experiment. AQuA remains the original explicit-letter score. Report tables show both protocols.

## Colab

Assignment2-Colab.ipynb and README-experiments.md document execution. A100/BF16 is the verified experimental profile. T4 full-FP32 is a candidate requiring its own smoke and memory checks; no T4 training results are claimed.

## Submission

Submit `AI6130-Ass2-G2605698H.pdf` to NTULearn Assignment 2 -> Report by 14 November 2026 23:59 Singapore time, using the details supplied for this assignment. The report includes the required AI use declaration. Only the report is the submission artifact here; the evidence package supports reproduction.
