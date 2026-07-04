# Idea Discovery Report

**Direction**: PAPI-style LLM decode acceleration with virtual PIM and dynamic scheduling  
**Date**: 2026-04-11  
**Pipeline**: research-lit -> idea-creator -> novelty-check -> research-review -> research-refine-pipeline

## Executive Summary

The original PAPI paper is an ASPLOS 2025 architecture paper whose main novelty is a **hardware-backed heterogeneous system** with **online kernel characterization** for dynamic scheduling across GPU and hybrid PIM units. A direct 1:1 reproduction is not realistic in a home setting because it depends on PIM hardware modeling and larger system simulation.

The strongest software-only direction is therefore **not** "reimplement PAPI," but to build a **trace-driven LLM decode profiling and scheduling framework** that identifies when decode kernels become memory-bound, attaches a calibrated virtual PIM cost model, and compares dynamic offloading policies. This preserves the spirit of PAPI while moving the contribution into a tractable systems-research space.

## Literature Landscape

### Anchor paper

- **PAPI: Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System** (ASPLOS 2025, arXiv:2502.15470)
- Core claims from the paper:
  - LLM decode kernels can dynamically shift between compute-bound and memory-bound regimes as decoding parallelism changes.
  - Static mapping of kernels to GPU vs. PIM is suboptimal.
  - A heterogeneous system with online bottleneck prediction and hybrid PIM units improves performance.
- Reported result: PAPI shows up to **1.8x** speedup over a heterogeneous baseline and **11.1x** over a PIM-only baseline.

### What the literature already covers

- **Hardware heterogeneity for LLM inference** is established. PAPI and adjacent heterogeneous serving work already argue that different phases or kernels prefer different hardware resources.
- **Long-context / KV-cache bottlenecks** are now central. Recent work on KV-cache-aware scheduling and long-context systems shows that memory pressure and bandwidth dominate decode as sequence length grows.
- **PIM for LLM inference** is increasingly active, especially for attention/KV-heavy paths and long-context decode.

### Structural gaps relevant to a home-implementable project

- Most PIM papers evaluate on **simulators or custom hardware assumptions**, not on a **reproducible software framework over real LLM traces**.
- There is still room for a **runtime decision framework** that answers:
  - Which decode kernels are memory-bound under which batch/context/speculation settings?
  - How stable is that classification across requests?
  - How much value comes from oracle scheduling vs. lightweight online policies?
- There is a gap between:
  - architecture papers that assume PIM hardware, and
  - production inference systems that expose real traces but do not reason about virtual PIM offload.

### Scope decision

The right scope is:

- **Do**: real LLM decode profiling, kernel/phase classification, virtual PIM cost modeling, policy comparison, sensitivity studies.
- **Do not**: claim a real PIM hardware contribution or full reproduction of the PAPI architecture.

## Ranked Ideas

### 1. PIMScope: Trace-Driven Virtual PIM Scheduler for LLM Decode — RECOMMENDED

- **Thesis**: Build a software framework that profiles real LLM decode execution, labels candidate kernels/regions as memory-bound vs. compute-bound, and evaluates dynamic scheduling policies under a virtual PIM cost model.
- **Why it fits**:
  - directly targets the core PAPI intuition,
  - does not require PIM hardware implementation,
  - produces a reusable research artifact,
  - can be evaluated with open-source models and standard GPUs.
- **Minimum deliverable**:
  - profiling harness in PyTorch/Transformers or vLLM,
  - kernel-region feature extraction,
  - virtual PIM latency/energy model,
  - policy simulator with static / heuristic / online / oracle baselines,
  - experiments over model size, batch size, context length, and decoding parallelism.
- **Pilot status**: no pilot run executed in this session because the local shell interface is failing; manual pilot is still feasible and should stay within a small-GPU budget with 7B-8B models.
- **Novelty assessment**: promising if framed as a **trace-driven runtime evaluation framework** rather than "yet another PIM scheduler."
- **Status**: RECOMMENDED

### 2. PIMMap: Decode-Phase Kernel Taxonomy and Transferability Study — BACKUP

- **Thesis**: Provide a systematic empirical study of which LLM decode kernels stay memory-bound across models, contexts, and serving modes, and which flip regime.
- **Contribution type**:
  - measurement paper / characterization paper,
  - strongest if it introduces a robust taxonomy and actionable thresholds.
- **Strength**:
  - easier than Idea 1,
  - lower implementation risk.
- **Weakness**:
  - likely weaker venue fit than a framework with policy evaluation.
- **Status**: BACKUP

### 3. PIMSpec: Speculative-Decoding-Aware Virtual PIM Scheduling — CONDITIONAL

- **Thesis**: Extend the framework to speculative decoding and show that assistant-model acceptance rates change which kernels should be offloaded.
- **Strength**:
  - closer to PAPI's "dynamic parallelism" story,
  - potentially more novel than simple batching-only studies.
- **Weakness**:
  - implementation complexity increases quickly,
  - harder to stabilize without a strong initial framework.
- **Status**: CONDITIONAL follow-on, not the first project cut.

## Eliminated Ideas

- **Full PAPI reproduction with heterogeneous PIM hardware model**
  - Eliminated because it needs hardware/simulator fidelity beyond a home setup.
- **Claiming a new accelerator architecture without implementation**
  - Eliminated because the evidence would be too weak.
- **Pure theoretical scheduler without real profiling**
  - Eliminated because the whole point is that decode bottlenecks shift dynamically in practice.

## Deep Novelty Verification

### Closest prior art

- **PAPI (ASPLOS 2025)** is the closest paper. It already owns:
  - the high-level motivation,
  - online kernel characterization,
  - dynamic scheduling across GPU and PIM,
  - hardware architecture contribution.
- Related recent works also cover:
  - heterogeneous LLM inference and scheduling,
  - long-context KV-cache bottlenecks,
  - PIM or near-memory acceleration for attention-heavy inference.

### What remains plausibly novel

- A **software-only, trace-driven evaluation framework** for PIM-style LLM decode scheduling on real runtimes.
- A **decision-layer study** focused on runtime observables:
  - profiler-derived features,
  - online classification robustness,
  - scheduling regret relative to oracle,
  - sensitivity to batching/speculation/context.
- A **calibrated virtual PIM methodology** that lets others test scheduler ideas without hardware simulation.

### Novelty risk

- If framed as "dynamic scheduling for PIM-accelerated LLM decode," novelty is weak because PAPI already does that.
- If framed as "the first reproducible trace-driven software framework for evaluating virtual PIM scheduling policies on real LLM decode workloads," novelty is materially stronger.

## External Critical Review

### Reviewer-style verdict

- **Overall score**: 7.8/10 if scoped as a framework paper; 4.5/10 if scoped as an architecture contribution.

### Main strengths

- Good problem choice: decode is genuinely bandwidth-sensitive and dynamic.
- Realistic execution plan for a small lab/home setup.
- Clear empirical story: profiling -> classification -> cost model -> scheduling policy comparison.

### Main weaknesses

- Risk of being perceived as "PAPI without hardware."
- Cost-model credibility can become the weakest link if calibration is shallow.
- If evaluation is only on one runtime and one model family, claims will look narrow.

### Minimum viable fixes

- Explicitly state that this is **not** a hardware paper and does **not** reproduce PAPI.
- Make the central artifact the **framework and methodology**.
- Include strong baselines:
  - GPU-only,
  - static offload,
  - heuristic online policy,
  - oracle scheduler.
- Include at least one stress axis where dynamic policy clearly matters:
  - context growth,
  - batch growth,
  - speculative decoding acceptance variability.

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Next Steps

- [ ] Implement the profiling harness over one runtime first (`Transformers` or `vLLM`)
- [ ] Launch the first measurement pass on 7B-class models
- [ ] Fit a simple virtual PIM cost model from bandwidth/latency assumptions
- [ ] Compare static / dynamic / oracle policies
- [ ] If signal is positive, extend to speculative decoding and longer context

## Sources

- PAPI paper PDF: https://people.mpi-sws.org/~cgiannoula/assets/publications/PAPI_asplos25_full.pdf
- PAPI arXiv metadata: https://arxiv.org/abs/2502.15470
- Online Scheduling for LLM Inference with KV Cache Constraints: https://www.microsoft.com/en-us/research/publication/online-scheduling-for-llm-inference-with-kv-cache-constraints/
- HETEGEN: https://arxiv.org/abs/2403.01164
- INFERCEPT: https://arxiv.org/abs/2402.01869
