# GPU Validation Summary

Date: 2026-04-12

## Purpose

Verify that the CPU-trace conclusions also appear on real GPU execution.

## GPU Runs Collected

### `gpt2`

| Setting | Attention Share | Online Speedup | Adaptive Speedup | Oracle Speedup |
|---|---:|---:|---:|---:|
| `ctx=256, bs=1` | 0.464 | 1.000 | 1.000 | 1.269 |
| `ctx=512, bs=1` | 0.466 | 1.143 | 1.296 | 1.296 |
| `ctx=768, bs=1` | 0.462 | 1.162 | 1.322 | 1.322 |
| `ctx=512, bs=2` | 0.459 | 1.141 | 1.296 | 1.296 |

### `opt-125m`

| Setting | Attention Share | Online Speedup | Adaptive Speedup | Oracle Speedup |
|---|---:|---:|---:|---:|
| `ctx=512, bs=1` | 0.852 | 1.298 | 1.348 | 1.348 |

## Immediate Interpretation

1. The weak-signal vs positive-signal split survives on GPU.
2. `ctx=256` remains a weak-signal regime for `gpt2`.
3. `ctx>=512` shows consistent positive dynamic-offload signal.
4. `opt-125m` is substantially more attention-heavy than `gpt2` in the current GPU runs.

## Important Caution

`adaptive_score` currently matches `oracle` on the collected positive-signal GPU traces.

This should **not** be interpreted as "the scheduling problem is solved."

Instead, it means the current prototype policy is too tightly coupled to the same internal cost-model signals used by the oracle. In other words, the policy is currently a strong prototype for exploring the decision space, but it is not yet a trustworthy final scheduling method.

## Recommended Next GPU Experiments

1. Add one more GPU point for `opt-125m`:
   - `ctx=256, bs=1`
2. Add one more GPU batch point:
   - `opt-125m, ctx=512, bs=2`
3. Add one more larger-context `gpt2` point if GPU permits:
   - `ctx=960, bs=1`
4. Re-test policies after making `adaptive_score` less oracle-like.
