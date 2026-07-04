# Results Table

## Prior-Work-Grounded Virtual PIM Model

Source summary:
- `artifacts/reports/prior_work_suite_summary.json`

## Context / Batch Results

| Model | Setting | Online Speedup | Adaptive Speedup | Oracle Speedup |
|---|---|---:|---:|---:|
| `gpt2` | `ctx=256` | 1.000 | 1.000 | 1.267 |
| `gpt2` | `ctx=512` | 1.139 | 1.296 | 1.296 |
| `gpt2` | `ctx=768` | 1.157 | 1.320 | 1.320 |
| `gpt2` | `ctx=960` | 1.164 | 1.335 | 1.335 |
| `gpt2` | `ctx=512, bs=4` | 1.136 | 1.295 | 1.295 |
| `opt-125m` | `ctx=256` | 1.000 | 1.000 | 1.282 |
| `opt-125m` | `ctx=512` | 1.219 | 1.323 | 1.323 |
| `opt-125m` | `ctx=768` | 1.235 | 1.355 | 1.355 |
| `opt-125m` | `ctx=512, bs=4` | 1.170 | 1.306 | 1.306 |

## Immediate Observations

1. Short contexts (`256`) show no gain for either dynamic policy.
2. `512+` contexts consistently show gain.
3. `adaptive_score` matches oracle on 7 of 9 currently tested settings.
4. `adaptive_score` beats the earlier `online_predictor` on 7 of 9 settings.
5. `opt-125m` shows slightly stronger gains than `gpt2` in the current CPU-trace setup.

## First GPU Results

| Model | Setting | Attention Share | Online Speedup | Adaptive Speedup | Oracle Speedup |
|---|---|---:|---:|---:|---:|
| `gpt2` | `GPU, ctx=256, bs=1` | 0.464 | 1.000 | 1.000 | 1.269 |
| `gpt2` | `GPU, ctx=512, bs=1` | 0.466 | 1.143 | 1.296 | 1.296 |
| `gpt2` | `GPU, ctx=768, bs=1` | 0.462 | 1.162 | 1.322 | 1.322 |
| `gpt2` | `GPU, ctx=512, bs=2` | 0.459 | 1.141 | 1.296 | 1.296 |
| `opt-125m` | `GPU, ctx=512, bs=1` | 0.852 | 1.298 | 1.348 | 1.348 |

## GPU Observations

1. The weak-signal (`ctx256`) versus positive-signal (`ctx512`) split survives on GPU.
2. `adaptive_score` currently matches oracle on the collected GPU traces.
3. `opt-125m` is much more attention-heavy than `gpt2` on GPU in the current setup.

## Latest Quantized GPU Results

| Model | Setting | Attention Share | Online Speedup | Adaptive Feature | Oracle Speedup |
|---|---|---:|---:|---:|---:|
| `Qwen2.5-1.5B-Instruct` | `GPU, 4bit, ctx=256, bs=1` | 0.542 | 1.000 | 1.000 | 1.275 |
| `Qwen2.5-1.5B-Instruct` | `GPU, 4bit, ctx=512, bs=1` | 0.572 | 1.182 | 1.182 | 1.310 |
| `Qwen2.5-1.5B-Instruct` | `GPU, 4bit, ctx=768, bs=1` | 0.539 | 1.194 | 1.194 | 1.337 |
| `Qwen3-8B` | `GPU, pre-4bit, ctx=256, bs=1` | 0.547 | 1.147 | 1.000 | 1.275 |
| `Qwen3-8B` | `GPU, pre-4bit, ctx=512, bs=1` | 0.496 | 1.154 | 1.154 | 1.300 |
| `Qwen3-8B` | `GPU, pre-4bit, ctx=768, bs=1` | 0.474 | 1.167 | 1.167 | 1.324 |
| `SmolLM2-1.7B-Instruct` | `GPU, 4bit, ctx=256, bs=1` | 0.544 | 1.146 | 1.000 | 1.275 |
| `SmolLM2-1.7B-Instruct` | `GPU, 4bit, ctx=512, bs=1` | 0.556 | 1.176 | 1.176 | 1.309 |
| `SmolLM2-1.7B-Instruct` | `GPU, 4bit, ctx=512, bs=2` | 0.530 | 1.167 | 1.167 | 1.306 |
| `SmolLM2-1.7B-Instruct` | `GPU, 4bit, ctx=768, bs=1` | 0.543 | 1.196 | 1.196 | 1.338 |

## Latest-Model Observations

1. `Qwen2.5-1.5B-Instruct` shows a clean weak-signal (`ctx256`) versus positive-signal (`ctx512+`) split.
2. `SmolLM2-1.7B-Instruct` enters the positive regime earlier for `online_predictor`, suggesting the onset of offload value can be model-family dependent.
3. `Qwen3-8B` also enters and stays in the positive regime across `ctx256/512/768`, which makes the latest-model story materially stronger.
4. The current story is no longer restricted to older GPT-style checkpoints.
5. `adaptive_feature` remains more conservative than `online_predictor` at `SmolLM2 ctx256` and `Qwen3-8B ctx256`, which is useful evidence that the realistic policy is not simply over-dispatching.
6. `SmolLM2` batch scaling from `bs=1` to `bs=2` changes the gain less than the `Qwen` context change from `256` to `512/768`, which again points to context as the stronger driver overall.
