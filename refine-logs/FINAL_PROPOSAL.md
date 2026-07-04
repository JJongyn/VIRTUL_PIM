# Final Proposal

## Problem Anchor

Can we build a **reproducible software framework** that identifies when real LLM decode regions are sufficiently memory-bound that a hypothetical PIM backend would help, and can we show that **online dynamic scheduling policies** outperform static policies under changing decode conditions?

## Method Thesis

Use real decode traces from open-source LLM runtimes, derive lightweight runtime features for each kernel/region, attach a calibrated virtual PIM cost model, and evaluate whether online scheduling closes a meaningful fraction of the gap to an oracle scheduler.

## Dominant Contribution

The dominant contribution is **methodology + framework**, not hardware:

- real trace collection on open-source LLM inference stacks,
- runtime bottleneck classification for decode regions,
- virtual PIM what-if evaluation,
- policy analysis under dynamic serving conditions.

## Why This Is Defensible

- It keeps the strongest idea from PAPI, dynamic kernel-to-device assignment.
- It removes the part that is not feasible at home, custom heterogeneous hardware evaluation.
- It creates a reusable research platform that others can extend to actual PIM simulators later.

## System Sketch

1. Run decoding workloads on a real stack (`Transformers` or `vLLM`).
2. Collect per-step and per-region measurements:
   - latency,
   - achieved FLOPs proxy,
   - bytes moved / KV-cache growth proxy,
   - sequence length,
   - batch size,
   - speculative depth or acceptance rate if applicable.
3. Split decode into scheduling units:
   - attention-related region,
   - FC / projection / MLP region,
   - optional KV-management region.
4. Predict whether each unit is:
   - GPU-favored,
   - PIM-favored,
   - ambiguous.
5. Evaluate policies:
   - GPU-only,
   - static attention-to-PIM,
   - threshold heuristic,
   - lightweight online predictor,
   - oracle.

## Main Hypotheses

- **H1**: Decode-region bottlenecks flip with context length, batch size, and decoding mode often enough that static mapping is measurably suboptimal.
- **H2**: A small online policy using runtime features can recover a substantial portion of oracle benefit.
- **H3**: The benefit is concentrated in attention/KV-heavy regimes, especially at longer contexts and larger effective decode parallelism.

## Claim Boundaries

We will claim:

- better scheduling decisions in a virtual-PIM what-if setting,
- a reproducible methodology for studying PIM-style decode scheduling.

We will not claim:

- a fabricated hardware speedup as real,
- superiority of a concrete PIM microarchitecture,
- full reproduction of PAPI.

## MVP

- One model family: Llama-2/3 or Qwen 7B-class
- One runtime first: `Transformers` or `vLLM`
- One calibrated virtual PIM model
- Four policies and one oracle
- Three stress dimensions: context, batch, decode mode

## Expansion Path

- Add speculative decoding
- Add multiple runtimes
- Add energy-aware objective
- Add KV-cache compression/offload interaction
