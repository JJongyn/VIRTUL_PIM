# Virtual PIM Experiment Log

Date: 2026-04-11

## Goal

Build and validate a trace-driven virtual PIM scheduling framework for LLM decoding, then check whether dynamic offload shows meaningful signal even in small-model CPU experiments.

## What Was Implemented

- Trace schema for decode-region records
- Transformers-based decode trace adapter
- Trace analysis utilities
- Virtual PIM cost model
- Policy simulator
- Context sweep runner
- Batch sweep runner

## Key Code Paths

- `src/papi_virtual/schema.py`
- `src/papi_virtual/adapters/transformers_decode.py`
- `src/papi_virtual/analysis.py`
- `src/papi_virtual/cost_model.py`
- `src/papi_virtual/policies.py`
- `src/papi_virtual/simulator.py`
- `scripts/run_cpu_signal_experiment.py`
- `scripts/run_context_sweep.py`
- `scripts/run_batch_sweep.py`

## Experiment Timeline

### 1. Minimal end-to-end smoke test

Purpose:
- Verify that the pipeline can run from trace collection to simulation.

Setup:
- model: `sshleifer/tiny-gpt2`
- device: `cpu`

Artifacts:
- `artifacts/traces/minimal_demo.jsonl`
- `artifacts/reports/minimal_trace_summary.json`
- `artifacts/reports/minimal_simulation.json`

Result:
- Pipeline worked end to end.
- This was only a framework smoke test, not a meaningful systems result.

### 2. First real small-model trace

Purpose:
- Replace synthetic trace with a real decode trace.

Setup:
- model: `distilgpt2`
- device: `cpu`
- short prompt / short context

Artifacts:
- `artifacts/traces/distilgpt2_cpu.jsonl`
- `artifacts/reports/distilgpt2_trace_summary.md`
- `artifacts/reports/distilgpt2_simulation.json`

Result:
- Trace collection succeeded.
- No offload signal appeared.
- Attention share was too small and context was too short.

Interpretation:
- Good negative result.
- Confirms that offload is not automatically beneficial in all settings.

### 3. Longer-context signal probe with `gpt2`

Purpose:
- Test whether longer context increases attention share and makes offload meaningful.

Setup:
- model: `gpt2`
- device: `cpu`
- prompt target: `512`
- max new tokens: `32`

Artifacts:
- `artifacts/traces/gpt2_cpu_signal.jsonl`
- `artifacts/reports/gpt2_cpu_signal_trace_summary.md`
- `artifacts/reports/gpt2_cpu_signal_simulation.json`
- `artifacts/reports/gpt2_cpu_signal_pim_sweep.json`

Initial Result:
- Attention share increased relative to very small runs.
- Static offload could be harmful or neutral under the early cost model.
- Parameter sweep showed positive oracle-only regions, meaning signal existed but the model/policy stack was not yet aligned.

### 4. Hook-based region timing and trace-calibrated cost model

Purpose:
- Improve trace fidelity.
- Stop relying only on coarse FLOP-based latency splitting.

Changes:
- Added module timing hooks for attention/MLP regions.
- Changed GPU baseline in the cost model to use observed trace latency.
- Scaled virtual PIM estimates relative to observed latency.

Artifacts:
- `artifacts/traces/gpt2_cpu_signal_hooked.jsonl`
- `artifacts/reports/gpt2_cpu_signal_hooked_trace_summary.md`
- `artifacts/reports/gpt2_cpu_signal_hooked_simulation_v2.json`
- `artifacts/reports/gpt2_cpu_signal_hooked_pim_sweep_v2.json`

Result:
- Attention share became much more realistic.
- For `gpt2 @ ~512 context`, results became:
  - `gpu_only`: `1.000x`
  - `static_attention`: `1.180x`
  - `online_predictor`: `1.180x`
  - `oracle`: `1.543x`

Interpretation:
- This was the first strong positive signal.
- Dynamic offload became meaningfully beneficial in the virtual PIM setting.

### 4.5. Prior-work-grounded PIM modeling

Purpose:
- Reduce arbitrary cost-model assumptions.
- Tie the virtual PIM backend more clearly to prior PIM-LLM papers.

Grounding choice:

- Use `PAPI` as the main reference for the GPU-side roofline assumptions and
  qualitative kernel behavior.
- Use `LoL-PIM` as a supporting reference for the observation that long
  context strengthens attention/KV memory pressure.

What was adopted:

- GPU default reference:
  - A100-like roofline parameters from `PAPI`
  - `312 TFLOPS`, `1935 GB/s`
- Region-level qualitative priors:
  - attention remains the strongest memory-centric offload candidate,
  - FC/MLP paths are relatively more compute-sensitive,
  - longer context favors memory-centric execution for attention/KV-like paths.

What remains virtual:

- The PIM backend is still a sweepable virtual backend, not a hardware-faithful
  cycle-accurate simulator.
- Therefore the model is still appropriate for:
  - trace-driven what-if studies,
  - policy comparison,
  - sensitivity analysis,
  but not for exact hardware-speedup claims.

### 4.6. Suite recomputation under prior-work-grounded assumptions

Purpose:
- Re-evaluate the main experiment suite after grounding the virtual backend in
  prior PIM-LLM work instead of freer earlier defaults.

Artifacts:

- `artifacts/reports/gpt2_ctx256_simulation_prior_work.json`
- `artifacts/reports/gpt2_ctx512_simulation_prior_work.json`
- `artifacts/reports/gpt2_ctx768_simulation_prior_work.json`
- `artifacts/reports/gpt2_ctx960_simulation_prior_work.json`
- `artifacts/reports/gpt2_ctx512_bs4_simulation_prior_work.json`
- `artifacts/reports/opt125m_ctx256_simulation_prior_work.json`
- `artifacts/reports/opt125m_ctx512_simulation_prior_work.json`
- `artifacts/reports/opt125m_ctx768_simulation_prior_work.json`
- `artifacts/reports/opt125m_ctx512_bs4_simulation_prior_work.json`
- `artifacts/reports/prior_work_suite_summary.json`

Headline result:

- Speedups became more conservative than before.
- The qualitative picture did not change.

Summary:

| Workload | Online | Adaptive | Oracle |
|---|---:|---:|---:|
| `gpt2 @ 256` | 1.000 | 1.000 | 1.267 |
| `gpt2 @ 512` | 1.139 | 1.296 | 1.296 |
| `gpt2 @ 768` | 1.157 | 1.320 | 1.320 |
| `gpt2 @ 960` | 1.164 | 1.335 | 1.335 |
| `gpt2 @ 512, bs=4` | 1.136 | 1.295 | 1.295 |
| `opt-125m @ 256` | 1.000 | 1.000 | 1.282 |
| `opt-125m @ 512` | 1.219 | 1.323 | 1.323 |
| `opt-125m @ 768` | 1.235 | 1.355 | 1.355 |
| `opt-125m @ 512, bs=4` | 1.170 | 1.306 | 1.306 |

Interpretation:

- Short context remains a weak-signal regime.
- `512+` context remains a positive-signal regime.
- The stronger `adaptive_score` prototype still closes the oracle gap on most
  tested positive regimes even under the more conservative, prior-grounded
  backend model.

### 5. Transfer check on `facebook/opt-125m`

Purpose:
- Check whether the signal generalizes across a second small model.

Setup:
- model: `facebook/opt-125m`
- device: `cpu`
- prompt target: `512`

Artifacts:
- `artifacts/traces/opt125m_cpu_signal_512_v2.jsonl`
- `artifacts/reports/opt125m_cpu_signal_512_v2_trace_summary.md`
- `artifacts/reports/opt125m_cpu_signal_512_v2_simulation.json`

Result:
- Attention share: `0.622`
- `gpu_only`: `1.000x`
- `static_attention`: `1.280x`
- `online_predictor`: `1.280x`
- `oracle`: `1.543x`

Interpretation:
- The dynamic-offload signal was not unique to `gpt2`.
- A second model showed the same direction, with even larger online gain.

### 5.5. First GPU trace checks

Purpose:
- Verify that the CPU-trace conclusions also appear on real GPU execution.

Artifacts:
- `artifacts/reports/gpt2_gpu_ctx256_bs1_trace_summary.md`
- `artifacts/reports/gpt2_gpu_ctx256_bs1_simulation.json`
- `artifacts/reports/gpt2_gpu_ctx512_bs1_trace_summary.md`
- `artifacts/reports/gpt2_gpu_ctx512_bs1_simulation.json`
- `artifacts/reports/gpt2_gpu_ctx768_bs1_trace_summary.md`
- `artifacts/reports/gpt2_gpu_ctx768_bs1_simulation.json`
- `artifacts/reports/gpt2_gpu_ctx512_bs2_trace_summary.md`
- `artifacts/reports/gpt2_gpu_ctx512_bs2_simulation.json`
- `artifacts/reports/opt125m_gpu_ctx512_bs1_trace_summary.md`
- `artifacts/reports/opt125m_gpu_ctx512_bs1_simulation.json`

Current GPU summary:

| Workload | Attention Share | Online | Adaptive | Oracle |
|---|---:|---:|---:|---:|
| `gpt2 @ GPU, ctx256, bs1` | 0.464 | 1.000 | 1.000 | 1.269 |
| `gpt2 @ GPU, ctx512, bs1` | 0.466 | 1.143 | 1.296 | 1.296 |
| `gpt2 @ GPU, ctx768, bs1` | 0.462 | 1.162 | 1.322 | 1.322 |
| `gpt2 @ GPU, ctx512, bs2` | 0.459 | 1.141 | 1.296 | 1.296 |
| `opt-125m @ GPU, ctx512, bs1` | 0.852 | 1.298 | 1.348 | 1.348 |

Interpretation:
- The same weak-signal / positive-signal split appears on GPU.
- `ctx256` remains a weak-signal regime.
- `ctx512` becomes a positive-signal regime on GPU as well.
- `adaptive_score` still matches oracle on the currently collected GPU traces.

## Context Sweep Results

### `gpt2`

Source:
- `artifacts/reports/gpt2_context_sweep.json`

Results:

| Context | Attention Share | Online Speedup | Oracle Speedup |
|---|---:|---:|---:|
| 256 | 0.426 | 1.000 | 1.543 |
| 512 | 0.453 | 1.190 | 1.543 |
| 768 | 0.449 | 1.188 | 1.543 |
| 960 | 0.434 | 1.180 | 1.543 |

Interpretation:
- At short context (`256`), dynamic offload gave no benefit.
- At `512+`, online gain consistently appeared.

### `facebook/opt-125m`

Source:
- `artifacts/reports/opt125m_context_sweep.json`

Results:

| Context | Attention Share | Online Speedup | Oracle Speedup |
|---|---:|---:|---:|
| 256 | 0.653 | 1.000 | 1.543 |
| 512 | 0.667 | 1.307 | 1.543 |
| 768 | 0.632 | 1.286 | 1.543 |

Interpretation:
- Same pattern as `gpt2`.
- Dynamic gain appeared once context became sufficiently large.

## Batch Sweep Results

### `gpt2 @ context 512`

Source:
- `artifacts/reports/gpt2_batch_sweep_ctx512.json`

Results:

| Batch | Attention Share | Online Speedup | Oracle Speedup |
|---|---:|---:|---:|
| 1 | 0.486 | 1.206 | 1.543 |
| 2 | 0.483 | 1.205 | 1.543 |
| 4 | 0.445 | 1.186 | 1.543 |

Interpretation:
- Increasing batch did not strengthen gain in this setup.
- The main useful driver so far is context length, not batch.

### `facebook/opt-125m @ context 512`

Source:
- `artifacts/reports/opt125m_batch_sweep_ctx512.json`

Results:

| Batch | Attention Share | Online Speedup | Oracle Speedup |
|---|---:|---:|---:|
| 1 | 0.616 | 1.277 | 1.543 |
| 2 | 0.568 | 1.250 | 1.543 |
| 4 | 0.540 | 1.235 | 1.543 |

Interpretation:
- Same tendency as `gpt2`.
- Higher batch slightly reduced attention share and online gain in the current CPU setup.

## New Policy Prototype: `adaptive_score`

### Motivation

The original `online_predictor` heuristic recovered part of the oracle gain but left a visible gap in the positive-signal regimes. While waiting for GPU experiments, a better policy was implemented to test whether the trace already contains enough information to support a stronger runtime dispatch rule.

### Policy Idea

`adaptive_score` combines several runtime signals:

- relative latency gain from sending a region to virtual PIM,
- memory-pressure proxy,
- arithmetic-intensity proxy,
- context length,
- batch size,
- a small region bias for attention-heavy regions.

The policy dispatches to PIM only when:

- context is sufficiently large, and
- estimated relative gain is positive enough, and
- the combined score passes a threshold.

Implementation:
- `src/papi_virtual/policies.py`

### First Evaluation

Artifacts:

- `artifacts/reports/gpt2_ctx512_simulation_adaptive.json`
- `artifacts/reports/gpt2_ctx960_simulation_adaptive.json`
- `artifacts/reports/opt125m_ctx512_simulation_adaptive.json`

Observed result:

| Workload | Online | Adaptive | Oracle |
|---|---:|---:|---:|
| `gpt2 @ 512` | 1.190 | 1.543 | 1.543 |
| `gpt2 @ 960` | 1.180 | 1.543 | 1.543 |
| `opt-125m @ 512` | 1.307 | 1.543 | 1.543 |

Interpretation:

- `adaptive_score` immediately closed the oracle gap on the first tested positive-signal traces.
- This suggested that the trace/cost-model combination already contains enough information to support a stronger dynamic rule than the original `online_predictor`.

### Holdout Check

Purpose:
- Verify that `adaptive_score` is not only fitting the first few hand-picked traces.

Artifacts:

- `artifacts/reports/gpt2_ctx256_simulation_adaptive.json`
- `artifacts/reports/gpt2_ctx768_simulation_adaptive.json`
- `artifacts/reports/gpt2_ctx512_bs4_simulation_adaptive.json`
- `artifacts/reports/opt125m_ctx256_simulation_adaptive.json`
- `artifacts/reports/opt125m_ctx768_simulation_adaptive.json`
- `artifacts/reports/opt125m_ctx512_bs4_simulation_adaptive.json`
- `artifacts/reports/adaptive_policy_holdout_summary.json`

Summary from holdout suite:

- Reports evaluated: `9`
- Cases where `adaptive_score` matched oracle: `7`
- Cases where `adaptive_score` beat the old `online_predictor`: `7`

Detailed takeaway:

- `adaptive_score` matched oracle on nearly all `512+` context cases tested so far.
- It did **not** close the oracle gap at the shortest `256` context cases.
- This is a useful pattern rather than a failure:
  - short context is a weak-signal regime,
  - longer context is a strong-signal regime,
  - the policy behaves differently across them.

Interpretation:

- The new policy is stronger than the original heuristic.
- It still does not prove broad generalization.
- At present it should be treated as a **strong prototype policy**, not yet as a fully validated final method.

## Current Takeaways

1. The framework works on real decode traces, not only synthetic traces.
2. Dynamic offload is not universally good; short contexts showed no gain.
3. Longer contexts consistently produced positive online speedup.
4. The trend appeared in both `gpt2` and `facebook/opt-125m`.
5. In the current experiments, context length is a stronger driver than batch size.
6. There is still a nontrivial oracle gap, so better policies remain possible.
7. A stronger feature-based prototype policy can close most of that gap on the currently tested positive-signal traces.
8. Re-grounding the backend in prior work reduced absolute speedups but preserved the main qualitative conclusions.
9. The first GPU traces preserve the same weak-signal / strong-signal pattern seen on CPU traces.

## Current Limits

- CPU-only experiments so far
- Small models only
- No speculative decoding yet
- Cost model is still lightweight
- Current online policy is heuristic, not learned

## Next Steps

1. Convert the current result files into paper-style figures/tables.
2. Re-test `adaptive_score` on GPU-backed traces when GPU access is available.
3. Add at least one GPU-backed trace run.
4. Add one more model family if feasible.
5. Decide whether to frame the paper mainly as:
   - a characterization/framework paper, or
   - a dynamic scheduling policy paper.
6. Expand GPU traces into a small context sweep.
