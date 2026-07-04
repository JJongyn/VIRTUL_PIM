# Reference Paper Summary

## Paper

**PAPI: Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System**  
ASPLOS 2025 / arXiv:2502.15470

## What The Paper Did

PAPI studies LLM decoding under **parallel decoding** settings such as batching and speculative decoding. Its main observation is that decode kernels do not remain permanently compute-bound or memory-bound; instead, their bottleneck can change dynamically as decoding parallelism changes at runtime.

To address this, the paper proposes:

- **online kernel characterization** to decide whether a kernel is compute-bound or memory-bound at runtime,
- a **PIM-enabled heterogeneous computing system** composed of GPU, host CPU, and hybrid PIM units,
- **dynamic scheduling** of kernels to the most suitable hardware unit.

The paper therefore contributes both:

- a **runtime scheduling policy**, and
- a **hardware/system architecture** for heterogeneous execution.

## Key Results

- PAPI reports large speedups over prior heterogeneous and PIM-only baselines.
- The core empirical message is that **static scheduling is often suboptimal** because decode parallelism changes online.

## What Is Hard To Reproduce At Home

- Detailed PIM hardware behavior
- Heterogeneous architecture co-design
- Full-system simulation fidelity
- Comparison against specialized PIM baselines

## What Can Be Retained In A Software-Only Project

- The core hypothesis that decode bottlenecks are **dynamic**
- The idea of **runtime classification** for scheduling decisions
- The need to compare:
  - static scheduling,
  - lightweight dynamic policies,
  - oracle scheduling

## Open Gaps / Improvement Directions

- Build a **real-trace, software-only framework** that can test PIM-style scheduling ideas without hardware implementation.
- Quantify **how often** decode regions flip regime across:
  - model size,
  - context length,
  - batch size,
  - speculative decoding behavior.
- Study the **quality of lightweight runtime features** for online classification.
- Expose the **regret gap** between heuristic policies and oracle scheduling.
- Provide a **virtual PIM methodology** that is reusable by others before they commit to simulator or hardware work.

## Recommended Positioning

Do not position the project as a reproduction of PAPI. Position it as:

**a software systems framework inspired by PAPI's dynamic scheduling insight, designed to evaluate virtual PIM offloading decisions on real LLM decode traces.**
