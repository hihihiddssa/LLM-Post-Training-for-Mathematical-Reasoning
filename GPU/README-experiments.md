# AI6130 Assignment 2

Student: AGA CILA, G2605698H. Upstream: https://github.com/duyngtr16061999/AI6130_Assignment2

The upstream files and bundled `peft/` remain in place. The supported experiment
commands below use the instructor's training/evaluation prompt functions and PEFT
classes, with corrected checkpointing, provenance and answer scoring. Read
`AUDIT.md` for the reasons for these changes.

Run the commands below from the sibling `GPU` directory. The teacher repository remains at `../AI6130_Assignment2`; generated runs and metadata stay here.

## Experiments

Two models x LoRA/AdapterP, each evaluated on AQuA and AddSub. `configs/main.json`
fixes two epochs, LR 3e-4, effective batch 4, micro batch 1, cutoff 256, validation
120, seed 42, no weight quantization. Both adapters use FP32 trainable weights
with BF16 base weights and automatic mixed precision on the main Ampere GPU profile. Micro batch 1 plus four
accumulation steps and gradient checkpointing reduce T4 memory use. Use the same
configuration on A40. The generated data manifest freezes identical train/val IDs
for all four runs. Output limits and scoring are identical across all comparisons.

AdapterP means the original `BottleneckConfig(use_adapterp=True)`, equivalent to
`--adapter_name bottleneck --use_adapterp`; it targets each Llama MLP's `down_proj`.
LoRA targets `q_proj` and `v_proj`, rank 8, alpha 16, dropout .05. Do not install
the public PEFT package over this fork. `tools/runtime.py` checks the actual import.

## Environment

Use Python 3.11 or 3.12. NSCC currently uses the existing Python 3.12.1 module and a separate environment that reads the existing PyTorch installation without changing it. On a Linux GPU compute allocation, create a new environment:

```bash
python3.11 -m venv .venv-ass2
source .venv-ass2/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-compatible.txt
python -m pip install --no-deps ../AI6130_Assignment2/peft
export HF_HOME="$PWD/.cache/huggingface"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python tools/audit.py
python tools/test_protocol.py
python tools/smoke.py
python tools/freeze_models.py
python -m pip freeze > artifacts/requirements-resolved.txt
```

The proposed Linux torch wheel needs a compatible NVIDIA driver; run `nvidia-smi`
on the allocated node first. Its exact Linux combination must pass smoke tests
there before formal training. The locally tested environment instead reuses
torch 2.11.0+cu128 on the Windows RTX 3060 with Transformers 4.38.0 and Accelerate
0.28.0; `artifacts/synthetic-smoke.json` records what actually passed. No external
commercial API or API key is used. A public Hugging Face model download needs
internet access only when files are not already cached.

On this Windows computer the isolated interpreter is
`D:\project\PEFT模型微调\GPU\.venv\Scripts\python.exe`. It inherits PyTorch from the
existing Anaconda-based Assignment 1 environment and installs its own older
Transformers dependencies. Do not update Assignment 1's environment.

## Smoke tests with the assigned models

Synthetic smoke validates the real adapter code without downloading a base model.
It is not fine-tuning an assigned model. Next run one actual training step, save,
checkpoint and evaluate on a small diagnostic subset for each model/method:

```bash
python tools/train.py --model tinyllama --method lora --smoke
python tools/train.py --model tinyllama --method adapterp --smoke
python tools/train.py --model openllama --method lora --smoke
python tools/train.py --model openllama --method adapterp --smoke
python tools/evaluate.py --run runs/smoke-tinyllama-lora-2ep --task AddSub --limit 2
```

Smoke directories are separate and never enter the main results table.

## Main training and evaluation

Run one process per allocated GPU, never on a login node:

```bash
python tools/train.py --model tinyllama --method lora
python tools/evaluate.py --run runs/main-tinyllama-lora-2ep --task AQuA
python tools/evaluate.py --run runs/main-tinyllama-lora-2ep --task AddSub
```

Repeat with methods `lora`, `adapterp` and models `tinyllama`, `openllama`.
`cluster/run_cell.sh MODEL METHOD` performs that sequence with independent logs.
Use the cluster's verified partition/QOS/GPU limits when submitting it. Do not
assume a partition or walltime based only on a template. `cluster/job.slurm` is a
starting point; the final submission command will record actual resource flags.

To resume a training interruption:

```bash
python tools/train.py --model tinyllama --method lora --resume
```

This loads the latest complete adapter+optimizer+scheduler+RNG checkpoint. To
pause gracefully, create `runs/main-tinyllama-lora-2ep/STOP_AFTER_STEP`; remove
that file before resuming. Completed training is detected and skipped. Training
and scoring source/config changes cause a refusal to mix runs. Preserve old
directories when adopting a new protocol.

Evaluation resumes by rerunning the same command, checks existing question IDs,
and appends durable JSONL rows. An interrupted final line is discarded and its
question rerun. Duplicate IDs or changed provenance cause an error. Save locally
on the GPU machine; periodically copy the run directory to persistent storage or
Drive rather than writing the full JSON file there after every question. The
generation change from four beams to greedy decoding is applied to all eight
evaluations, with KV cache enabled and score tensors disabled. Prompt text is
unchanged. Numeric tolerance remains the instructor's absolute 0.001.

## Results and report

```bash
python tools/summarize.py
```

This rechecks gold labels, saved predictions and scores and writes CSV, JSON and
Markdown. Accuracy stays empty until every question in that cell is present.
Each of four rows has prepared/trained/evaluated/summarized states. Raw outputs,
invalid/truncated/repeated output diagnostics, hardware and trainable parameters
remain in its own run directory. Test `output` rationales are never passed to a
model or used as the gold label. `report/AI6130-Ass2-G2605698H-draft.md` is a draft
until the required experiments complete. No results are invented.

The earlier Colab OpenLLaMA+LoRA one-epoch run is a separate configuration. If its
adapter is recovered from Drive, preserve and inspect its metadata before reuse.
An AddSub log ending at 13/395 is incomplete. Do not treat it as a benchmark score
or merge it into a two-epoch run.

Submission: report only, NTULearn Assignment 2 > Report, filename
`AI6130-Ass2-G2605698H`, due 14 November 2026 at 23:59 (Singapore time), according
to the user-provided assignment requirements. Keep the final English body within
four pages excluding cover. Include an honest AI-use declaration. Use direct
results-led prose and discuss concrete findings, with no defensive boilerplate.

## Current NSCC deployment

The current six PBS job IDs are in `artifacts/pbs-submission.json`; their initial state is summarized in `NSCC-status.md`. The release is under `/scratch/users/ntu/cila001/AI6130-Ass2-G2605698H/releases/20261101-v1/GPU`, also reachable through the per-project home symlink. Preparation installs dependencies and caches the two models; the GPU smoke job validates both methods on both models and checks checkpoint resumption. All four main jobs require that smoke job to succeed. Each main job performs training and both complete evaluations.

The home allocation was already above its 50GB quota. Only this assignment's failed upload directory was moved to scratch; other projects were preserved. Package caches, temporary files, model downloads and new adapters stay under the assignment scratch path.

BF16 was selected using fixed training-item gradient checks, not benchmark scores. The `configs/t4-fp32.json` fallback uses full FP32 and must pass its own smoke tests on a T4; its memory fit has not been verified. The old FP16 candidate and failed diagnostic runs are retained as evidence.

For Colab use `Assignment2-Colab.ipynb` and upload the workflow archive from `deploy/`. The report draft remains pending until the four main runs and eight full evaluations complete.
