# Backend Formalization

## Latency Model

- GPU latency estimate:
  - `L_gpu(r) = latency_ms(r)` when the trace already contains a measured region latency.
  - Otherwise `L_gpu(r) = max(FLOPs / TFLOPS_gpu, Bytes / BW_gpu)`.
- PIM latency estimate:
  - `L_pim(r) = max(L_gpu * compute_share * TFLOPS_gpu / TFLOPS_pim_eff, L_gpu * memory_share * BW_gpu / BW_pim_eff) + transfer + sync`.
- Effective modifiers:
  - Attention long-context bandwidth bonus: `0.20` scaled by `min(context / 1024, 1)`.
  - MLP parallelism penalty: `0.25` scaled by `min(decode_parallelism / 32, 1)`.
  - Non-attention throughput penalty: `0.10`.

## Provenance Split

- Trace-measured: region labels, measured latency, context length, batch size.
- Trace-derived: bytes moved, FLOPs, arithmetic intensity, estimated KV bytes, KV bytes per token, memory-pressure proxy.
- Modeled only: roofline compute/memory shares, effective PIM bandwidth/throughput, transfer and sync overheads, estimated PIM latency.

## Calibration Anchor

- The GPU side is anchored to trace-measured kernel latency whenever the trace provides latency_ms; the roofline terms are used only to decompose that measured latency into compute- and memory-dominant shares before applying the virtual PIM scaling.

## Parameter Source Notes

- GPU roofline anchor (A100 312 TFLOPS / 1935 GB/s) follows the prior-work setup referenced in the code comments and paper discussion of PAPI-style grounding.
- PIM bandwidth is set above the GPU anchor (2400 vs 1935 GB/s) to model a memory-centric accelerator advantage without assuming GPU-class compute throughput.
- PIM throughput is intentionally low (18 TFLOP/s) to encode a conservative memory-centric backend and to avoid overstating gains on compute-dominant regions.
- Attention long-context bandwidth bonus and MLP/non-attention penalties are qualitative priors used to encode the intended memory-vs-compute asymmetry, not hardware-calibrated constants.
- The attention bonus rewards long-context memory sensitivity, while FC and non-attention penalties are deliberately chosen to avoid overestimating benefit outside attention-heavy regions.
- Transfer and sync overheads are explicit constant terms and are varied again in the sensitivity study rather than treated as fixed hardware facts.

## Interpretation Boundary

- This backend is suitable for trace-grounded replay, signal comparison, and sensitivity analysis.
- It is not a cycle-accurate simulator and should not be used to claim deployment-ready hardware speedups.
- The paper should therefore treat backend-derived policy differences as evidence about predictive decision signals under the current model, not as universal scheduler truths.
