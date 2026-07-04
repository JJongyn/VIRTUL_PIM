# Virtual PIM Paper Readout

Date: 2026-04-12

## One-Sentence Summary

We built a trace-driven virtual PIM scheduling framework for LLM decoding and found a consistent pattern across CPU, GPU, and first modern 4-bit quantized instruct models: short-context decode is a weak-signal regime, while longer-context decode produces meaningful dynamic-offload gain, especially for attention-heavy models.

## Original Goal

The original goal was **not** to reproduce a real PIM hardware system.  
The goal was to test the following systems hypothesis:

1. LLM decode contains regions whose bottleneck shifts with runtime conditions.
2. Some decode regions become sufficiently memory-centric that a PIM-like backend could help.
3. Static offload rules are not enough.
4. A dynamic runtime policy can recover meaningful benefit in a virtual PIM setting.

## What Was Built

### Framework

- Trace schema for decode-region logging
- Transformers-based decode trace adapter
- Region timing hooks for attention/MLP
- Trace analysis utilities
- Virtual PIM cost model
- Policy simulator
- Context sweep runner
- Batch sweep runner
- GPU validation path

### Key Code

- `src/papi_virtual/schema.py`
- `src/papi_virtual/adapters/transformers_decode.py`
- `src/papi_virtual/analysis.py`
- `src/papi_virtual/cost_model.py`
- `src/papi_virtual/policies.py`
- `src/papi_virtual/simulator.py`

## Cost Model Positioning

The current backend is a **prior-work-grounded virtual PIM model**, not a cycle-accurate hardware simulator.

Grounding:

- `PAPI` for the GPU-side roofline reference and qualitative attention-vs-FC behavior
- `LoL-PIM` for the long-context intuition that attention/KV memory pressure increases with context length

What this means:

- Good for trace-driven what-if evaluation
- Good for policy comparison
- Good for sensitivity studies
- Not sufficient for direct hardware-speedup claims

## Policies Evaluated

- `gpu_only`
- `static_attention`
- `threshold`
- `online_predictor`
- `adaptive_feature`
- `adaptive_score`
- `oracle`

### Important distinction

- `adaptive_score` is a strong prototype but too oracle-like because it is still tightly coupled to cost-model internals.
- `adaptive_feature` is the more realistic candidate policy because it uses only runtime-style features such as:
  - context length
  - batch size
  - bytes moved
  - arithmetic intensity
  - latency
  - region bias
- `adaptive_family` is a follow-up realistic policy that adds only model-family priors on top of runtime features.

## Main Experimental Findings

### 1. Short context is a weak-signal regime

This pattern appears on both CPU and GPU.

Examples:

- `gpt2 @ CPU, ctx256`: `online = 1.000x`, `oracle = 1.267x`
- `gpt2 @ GPU, ctx256`: `online = 1.000x`, `oracle = 1.269x`
- `opt-125m @ CPU, ctx256`: `online = 1.000x`, `oracle = 1.282x`
- `opt-125m @ GPU, ctx256`: `online = 1.000x`, `oracle = 1.295x`

Interpretation:

- Offload value is not automatic.
- The framework is not producing fake gains everywhere.

### 2. `ctx >= 512` is consistently a positive-signal regime

This is the strongest recurring pattern in the current experiments.

Examples:

- `gpt2 @ CPU, ctx512`: `online = 1.139x`, `oracle = 1.296x`
- `gpt2 @ GPU, ctx512, bs1`: `online = 1.143x`, `oracle = 1.296x`
- `gpt2 @ GPU, ctx768, bs1`: `online = 1.162x`, `oracle = 1.322x`
- `gpt2 @ GPU, ctx960, bs1`: `online = 1.177x`, `oracle = 1.341x`
- `opt-125m @ CPU, ctx512`: `online = 1.219x`, `oracle = 1.323x`
- `opt-125m @ GPU, ctx512, bs1`: `online = 1.298x`, `oracle = 1.348x`
- `pythia-160m @ CPU, ctx512`: `online = 1.109x`, `oracle = 1.284x`
- `pythia-160m @ GPU, ctx512`: `online = 1.173x`, `oracle = 1.307x`

Interpretation:

- Dynamic offload gain appears repeatedly once context becomes large enough.
- This is currently the clearest positive result in the project.

### 3. More attention-heavy models tend to show larger gains

Examples:

- `gpt2 @ GPU, ctx512, bs1`
  - attention share `0.466`
  - online `1.143x`
- `opt-125m @ GPU, ctx512, bs1`
  - attention share `0.852`
  - online `1.298x`
- `pythia-160m @ GPU, ctx512, bs1`
  - attention share `0.549`
  - online `1.173x`

Interpretation:

- The current data supports the claim that more attention-heavy workloads are more favorable for virtual PIM offload.

### 4. The pattern survives on newer quantized instruct models

We added newer, more realistic small open-weight models in 4-bit form on GPU:

- `Qwen/Qwen2.5-1.5B-Instruct @ GPU, 4bit, ctx256`
  - attention share `0.542`
  - online `1.000x`
  - adaptive_feature `1.000x`
  - oracle `1.275x`
- `Qwen/Qwen2.5-1.5B-Instruct @ GPU, 4bit, ctx512`
  - attention share `0.572`
  - online `1.182x`
  - adaptive_feature `1.182x`
  - oracle `1.310x`
- `Qwen/Qwen2.5-1.5B-Instruct @ GPU, 4bit, ctx768`
  - attention share `0.539`
  - online `1.194x`
  - adaptive_feature `1.194x`
  - oracle `1.337x`
- `unsloth/Qwen3-8B-unsloth-bnb-4bit @ GPU, ctx256`
  - attention share `0.547`
  - online `1.147x`
  - adaptive_feature `1.000x`
  - oracle `1.275x`
- `unsloth/Qwen3-8B-unsloth-bnb-4bit @ GPU, ctx512`
  - attention share `0.496`
  - online `1.154x`
  - adaptive_feature `1.154x`
  - oracle `1.300x`
- `unsloth/Qwen3-8B-unsloth-bnb-4bit @ GPU, ctx768`
  - attention share `0.474`
  - online `1.167x`
  - adaptive_feature `1.167x`
  - oracle `1.324x`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct @ GPU, 4bit, ctx512`
  - attention share `0.556`
  - online `1.176x`
  - adaptive_feature `1.176x`
  - oracle `1.309x`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct @ GPU, 4bit, ctx256`
  - attention share `0.544`
  - online `1.146x`
  - adaptive_feature `1.000x`
  - oracle `1.275x`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct @ GPU, 4bit, ctx512, bs2`
  - attention share `0.530`
  - online `1.167x`
  - adaptive_feature `1.167x`
  - oracle `1.306x`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct @ GPU, 4bit, ctx768`
  - attention share `0.543`
  - online `1.196x`
  - adaptive_feature `1.196x`
  - oracle `1.338x`

Interpretation:

- The current story is no longer limited to older GPT-style checkpoints.
- `Qwen2.5-1.5B-Instruct` now shows the same weak-signal (`ctx256`) versus positive-signal (`ctx512+`) split seen earlier on older models.
- `Qwen3-8B` stays in the positive regime across `ctx256/512/768`, which materially strengthens the “latest model” story.
- `SmolLM2` appears to enter the positive regime earlier for `online_predictor`, which suggests the onset of offload value can be model-family dependent.
- `SmolLM2` also shows only modest change from `bs1` to `bs2` at `ctx512`.
- This makes the paper story more current and more defensible.

### 5. Batch is a weaker driver than context in the current setup

Examples:

- `gpt2 @ CPU, ctx512`
  - `bs1`: `1.206x`
  - `bs2`: `1.205x`
  - `bs4`: `1.186x`
- `opt-125m @ CPU, ctx512`
  - `bs1`: `1.277x`
  - `bs2`: `1.250x`
  - `bs4`: `1.235x`

Interpretation:

- Batch does matter, but current results suggest that context is the stronger explanatory variable.

## KV-Cache Mechanism Readout

Source:
- `artifacts/reports/KV_MECHANISM_REPORT.md`
- `artifacts/reports/kv_mechanism_dataset.csv`
- `figures/fig4_context_vs_kvbytes.pdf`
- `figures/fig5_kvbytes_vs_speedup.pdf`

We refreshed a focused GPU suite with explicit KV-cache proxy fields in the trace metadata:

- models: `gpt2`, `Qwen2.5-1.5B-Instruct 4bit`, `Qwen3-8B 4bit`, `SmolLM2-1.7B-Instruct 4bit`
- contexts: `256 / 512 / 768`
- extra batch point: `SmolLM2 @ ctx512, bs2`

Current focused-GPU mechanism correlations over `13` points:

- `context vs mean_kv_bytes_est = 0.509`
- `context vs online_speedup = 0.691`
- `mean_kv_bytes_est vs online_speedup = 0.585`
- `mean_kv_bytes_per_token_est vs online_speedup = 0.402`
- `memory_pressure_proxy vs online_speedup = 0.733`
- `mean_kv_bytes_est vs oracle_speedup = 0.607`
- `memory_pressure_proxy vs oracle_speedup = 0.935`

Interpretation:

- The long-context story is now better grounded than before.
- We no longer only observe `context -> speedup`; we now also observe `context -> KV traffic` and `KV traffic -> speedup`.
- This is still a proxy-based argument, not a hardware-counter result, but it is much closer to the actual systems mechanism than context alone.
- Family differences remain visible even under similar contexts, which supports the model-dependent onset story.

### KV-aware realistic policy follow-up

We also added a new realistic policy, `kv_regime`, that uses only:

- `context`
- `kv_bytes_est`
- `kv_bytes_per_token_est`
- `memory_pressure_proxy`
- basic runtime features such as latency and intensity

It does **not** use direct cost-model internals such as `gpu_est` / `pim_est`.

Focused-suite readout:

- `kv_regime` beats `adaptive_feature` on `2/13` focused points
- `kv_regime` ties `adaptive_feature` on `10/13`
- the two meaningful wins are:
  - `Qwen3-8B @ ctx256`
  - `SmolLM2-1.7B @ ctx256`

Interpretation:

- A KV-aware realistic policy can recover early positive-regime cases that the generic `adaptive_feature` policy misses.
- This strengthens the claim that family-dependent onset is connected to KV-pressure differences rather than to arbitrary heuristic tuning.

## Correlation Readout

Source:
- `artifacts/reports/CORRELATION_REPORT.md`

Current correlations over `27` CPU/GPU points:

- attention_share vs online_speedup: `0.272`
- attention_share vs adaptive_feature_speedup: `0.253`
- attention_share vs adaptive_family_speedup: `0.323`
- attention_share vs oracle_speedup: `0.321`

Interpretation:

- The correlation remains positive after adding modern quantized-model points.
- It is still not strong enough to be the paper's only evidence.
- It supports the story, but the newer KV-cache mechanism study is now the stronger complementary evidence.

## Regime Readout

Source:
- `artifacts/reports/REGIME_INSIGHT_REPORT.md`

Current online-positive regime summary using `online_speedup >= 1.05`:

- total points: `27`
- online-positive points: `22`
- online-weak points: `5`
- mean context in online-positive regime: `599.3`
- mean context in online-weak regime: `256.0`
- mean attention share in online-positive regime: `0.545`
- mean attention share in online-weak regime: `0.588`

Interpretation:

- `context` is currently the clearest separator between weak and positive regimes.
- `attention share` alone is not enough to define the regime boundary.
- This supports a stronger paper message:
  - positive offload value depends on **context-sensitive regime entry**
  - not on a single static attention heuristic

## Policy Ablation Readout

Source:
- `artifacts/reports/adaptive_feature_ablation.csv`

Average oracle-gap-closed across the ablation set:

- `full`: `0.642`
- `no_memory_pressure`: `0.640`
- `no_context`: `0.000`
- `no_batch`: `0.640`
- `no_bytes`: `0.192`
- `no_intensity`: `0.192`
- `no_latency`: `0.192`
- `no_region_bias`: `0.184`

Interpretation:

- `context` is the most important feature in the current realistic policy.
- `bytes`, `intensity`, `latency`, and `region bias` are also important.
- `batch` and `memory_pressure` appear less critical in the current dataset.

## Realistic Policy Status

The realistic policy story is now split into two layers:

- `adaptive_feature`
  - conservative
  - stable across most collected traces
  - still the safest realistic policy candidate
- `adaptive_family`
  - adds model-family-aware priors
  - successfully recovers the `SmolLM2 ctx256` case that `adaptive_feature` misses
  - but does not yet improve the full suite on average

Current aggregate readout over `24` points:

- `adaptive_family` beats `adaptive_feature` on `1` point
- `adaptive_family` is worse on `3` points
- `adaptive_family` ties on `20` points
- mean oracle-gap-closed
  - `adaptive_feature`: `0.478`
  - `adaptive_family`: `0.458`

Interpretation:

- Model-family-aware priors are a valid next direction.
- The current `adaptive_family` prototype is evidence that model-family-dependent positive-regime onset exists.
- It is not yet the final realistic policy.

## Current Best Reading Of The Hypothesis

### What the results support now

1. The decode offload problem is condition-dependent, not fixed.
2. Context length is the clearest driver of virtual PIM usefulness in the current study.
3. Static attention offload is useful but consistently suboptimal.
4. Dynamic policies are justified.
5. The overall pattern survives the move from CPU traces to first GPU traces.
6. The onset of positive offload value can be model-family dependent, especially on newer quantized instruct models.
7. The context effect is now partially explained by explicit KV-cache and memory-pressure proxies.

### What the results do not support yet

1. Real hardware speedup claims
2. Broad generalization to larger serving models
3. A final deployment-grade online policy
4. Strong claims about batch/parallelism behavior across realistic serving regimes

## What Is Strong Enough For A Paper

Already strong enough:

- Framework contribution
- Trace-driven methodology
- Weak-signal vs positive-signal regime split
- CPU-to-GPU qualitative consistency
- Multiple model families showing similar direction
- Initial modern quantized-model evidence
- Policy ablation evidence
- Initial model-family-aware policy evidence
- Initial KV-cache / memory-pressure mechanism evidence

Not yet strong enough:

- Final policy contribution
- Broad GPU scaling story
- Large-model validation
- Serving-stack diversity

## Recommended Paper Framing Right Now

The safest current framing is:

**A trace-driven framework and empirical study for virtual PIM scheduling in LLM decoding, showing that offload value emerges in longer-context, attention-heavy regimes and is consistent with increased KV-cache-driven memory pressure, while dynamic policies outperform static rules.**

This is stronger and safer than claiming a final scheduling method.

## Most Important Remaining Experiments

1. One more KV-aware realistic policy if time allows
2. Tighten the discussion around why batch is weaker than context in the current setup
3. Write the paper around the mechanism story instead of chasing many more models
4. Optional: one more larger modern quantized model only if resources allow

## Pointers

- Progress log:
  - `experiment-notes/2026-04-11_virtual_pim_progress/README.md`
- Tables:
  - `experiment-notes/2026-04-11_virtual_pim_progress/RESULTS_TABLE.md`
- GPU summary:
  - `experiment-notes/2026-04-11_virtual_pim_progress/GPU_VALIDATION.md`

## 2026-04-13 Follow-Up Update

After the initial 2026-04-12 readout, we ran one more modern-family validation on `microsoft/Phi-3.5-mini-instruct 4bit` and added a mid-context boundary point.

### `Phi-3.5` readout

- `ctx256`
  - attention share `0.517`
  - online `1.140x`
  - adaptive_feature `1.000x`
  - kv_regime `1.140x`
  - oracle `1.276x`
- `ctx384`
  - attention share `0.609`
  - online `1.181x`
  - adaptive_feature `1.181x`
  - kv_regime `1.181x`
  - oracle `1.297x`
- `ctx768`
  - attention share `0.492`
  - online `1.175x`
  - adaptive_feature `1.175x`
  - kv_regime `1.175x`
  - oracle `1.329x`

Interpretation:

- `Phi-3.5` behaves like another early-positive modern family rather than a delayed-onset family.
- The new `ctx384` point removes the coarse gap between `256` and `768` and shows that the early-positive result is not a one-off short-context artifact.
- `kv_regime` again recovers the early-positive `ctx256` case that `adaptive_feature` misses.

### `Phi-3.5` robustness readout

Source:
- `artifacts/reports/PHI35_COST_SENSITIVITY_REPORT.md`
- `artifacts/reports/PHI35_COST_SENSITIVITY_SUMMARY.md`

Current `Phi-3.5` sensitivity:

- `ctx256`
  - conservative: `1.046x`
  - default: `1.140x`
  - optimistic: `1.219x`
- `ctx768`
  - conservative: `1.087x`
  - default: `1.175x`
  - optimistic: `1.247x`

Interpretation:

- The long-context positive result remains stable under all three presets.
- The short-context `ctx256` point is an early-onset case that remains positive but sits closer to the decision boundary under the conservative preset.
- This is consistent with the current main story:
  - long-context positive regimes are robust,
  - some family-specific early-onset cases sit closer to the boundary.
