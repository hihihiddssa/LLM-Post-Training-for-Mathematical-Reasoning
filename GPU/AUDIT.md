# Code and data audit

Upstream commit: `21086af375ddb6433e8dd00df0a28eeb5ce65d25`.

| Finding | Implementation / impact |
| --- | --- |
| AQuA takes the first occurrence of A/B/C/D/E anywhere | New versioned parser requires a standalone label or explicit final-answer signal; conflicts and unresolved option lists are invalid. May change scores relative to the original script. |
| AddSub takes the last number in the whole response | New parser uses an explicit final numerical answer, boxed answer, #### answer or a numeric-only response. No fallback to the last reasoning number. Absolute tolerance stays 0.001. Scores are generative final-answer accuracy. |
| Evaluation decodes input+output and splits a response delimiter | Decode only generated token IDs, using the original prompt. Prevents prompt labels/embedded delimiters becoming predictions. |
| Evaluation disables KV cache and keeps beam scores | Enable cache, do not return unused scores; uniformly use greedy generation with max_new_tokens=256 instead of four beams. This changes decoding for every cell, explicitly recorded. |
| Whole result JSON is rewritten after every item; no resume | Append durable per-item JSONL; validate identities, IDs, gold labels and scores when resuming. Keep local compute storage as the active writer. |
| Original model CLI choices omit assigned models; output labels may collide | Separate explicit model keys and independent run directories, each with frozen config, model revision, data and code hashes. |
| sys.path.append can import installed PEFT first | Insert the instructor peft/src path first and assert the imported module path and BottleneckConfig. |
| lora_target_modules parameter is printed but unused | Driver explicitly configures the teacher LoRA q_proj/v_proj targets. AdapterP still uses teacher bottleneck+use_adapterp mapping to down_proj. |
| Seed is not applied before adapter initialization; shuffle seeds omitted | Seed before model/adapter initialization; freeze one split (6,731 train / 120 validation) and Trainer data seed 42 for every run. Split IDs differ from upstream datasets.train_test_split but are shared across all new runs. |
| Original gradient checkpointing helper acts only for INT8 | Explicitly enable non-reentrant checkpointing and input gradients for FP16. |
| Original state_dict monkey-patch and old PEFT can fail Trainer best-model/restart loading | AdapterTrainer saves the original fork's adapter format and explicitly restores adapter keys for optimizer-checkpoint resumption and best-validation-loss selection. Optimizer, scheduler, RNG and Trainer state remain Trainer-managed. |
| Original automatic torch.compile occurs after Trainer construction | No compile in the reproducible driver. |
| requirements.txt specifies Transformers 4.49 / Accelerate 1.5.2, unlike Colab | Isolated Colab-era Transformers 4.38 / Accelerate .28 pins; actual tested hardware/versions saved. Synthetic GPU smoke passed both teacher adapters locally. Linux/assigned-model smoke must also pass. |
| Dataset name math_7k does not mean exactly 7,000 items | Actual 6,851 records. AQuA test 254, AddSub test 395. All answer keys left untouched. Exact whitespace-normalized train/test instruction overlap was zero. |
| Some test rationale/output fields disagree with answer | Gold always comes from answer; evaluation prompt uses instruction/input only. Do not "correct" benchmark keys using generated rationales. |

Training keeps the teacher's full-prompt language-model objective (train_on_inputs
= true), including its prompt whitespace. Truncation at 256 tokens may remove
the final solution from long training items; tokenization records how many items
reach the cap. Do not select hyperparameters or parser rules to maximize test
accuracy. Parser tests use synthetic examples. Preserve raw model responses.

Offline tests: first-capital AQuA trap; final-number AddSub trap; conflicting
signals; fractions/negative/commas; incomplete reasoning; numeric tolerance;
duplicate/changed-identity rejection; interrupted-line recovery; incomplete-cell
accuracy suppression. The adapter smoke asserts nonzero gradients, updated
adapter tensors, unchanged base weights, matching logits after reloading and
working generation for both methods.

## Precision validation

On four fixed training items, TinyLlama FP16 forward losses were finite but gradients were not. BF16 gradients were finite for all four. The true BF16 TinyLlama+LoRA single-step test changed adapter weights and saved/reloaded successfully. Main runs therefore use BF16 uniformly. Training pads on the right; evaluation retains left padding. All failed FP16 smoke runs are preserved as diagnostics, never main results.
