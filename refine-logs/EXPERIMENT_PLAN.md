# Experiment Plan

## Goal

Validate that a trace-driven virtual PIM scheduler for LLM decoding is both empirically meaningful and publishable as a systems methodology paper.

## Experimental Blocks

### Block 1. Decode Profiling Baseline

- **Question**: Which decode regions are memory-bound vs. compute-bound under realistic operating points?
- **Setup**:
  - model: 7B-class decoder-only LLM
  - runtime: `Transformers` or `vLLM`
  - factors:
    - batch size: 1, 2, 4, 8
    - prompt length: 512, 2k, 8k
    - generated length: 128, 512
- **Outputs**:
  - per-region latency breakdown,
  - memory-pressure proxy,
  - regime map over batch/context settings.

### Block 2. Virtual PIM Cost Model

- **Question**: Under which bandwidth/latency assumptions would offloading help?
- **Method**:
  - define PIM service cost = transfer/setup + in-memory compute + synchronization
  - sweep plausible bandwidth and launch-overhead ranges
  - calibrate against known relative bandwidth advantages reported in PIM literature
- **Outputs**:
  - sensitivity curves,
  - feasible operating region for PIM benefit.

### Block 3. Policy Comparison

- **Question**: Does dynamic scheduling beat static scheduling?
- **Policies**:
  - GPU-only
  - static attention-to-PIM
  - threshold heuristic
  - online predictor
  - oracle
- **Metrics**:
  - end-to-end decode latency,
  - normalized throughput,
  - regret to oracle,
  - number of wrong dispatches.

### Block 4. Dynamic Parallelism Stress Test

- **Question**: When does static scheduling fail most badly?
- **Stress axes**:
  - continuous batching or changing active batch size,
  - growing context length,
  - optional speculative decoding with varying acceptance rate.
- **Outputs**:
  - cases where bottleneck regime flips,
  - policy robustness plots.

### Block 5. Generalization

- **Question**: Do conclusions transfer across models or runtimes?
- **Setup**:
  - second model family or second runtime
- **Outputs**:
  - transferability analysis,
  - limitations.

## Required Baselines

- GPU-only
- Static offload
- Oracle dispatch
- At least one lightweight online policy

## Metrics

- Latency per generated token
- Tokens/s
- P50/P95 latency
- Regret to oracle
- Fraction of decode units labeled memory-bound
- Estimated energy proxy if available

## Run Order

1. Profiling harness on a single model/runtime.
2. Regime map over context and batch.
3. Cost-model sweep.
4. Static vs. dynamic vs. oracle policy comparison.
5. One transferability experiment.

## Success Criteria

- A nontrivial fraction of decode regions flip preferred device across settings.
- Dynamic policy recovers a meaningful share of oracle advantage.
- The result holds on more than one workload shape.

## Failure Conditions

- Bottleneck classification is too stable, making dynamic scheduling unnecessary.
- Cost-model conclusions are too sensitive to arbitrary assumptions.
- Gains only appear in one narrow corner case.

## Compute Budget

- Initial profiling should fit on a single consumer GPU with 7B-class models.
- Keep first pass under 1 GPU-day.
- Defer speculative decoding and second-runtime expansion until initial signal is positive.
