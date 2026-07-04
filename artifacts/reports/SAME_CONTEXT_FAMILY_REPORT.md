# Same-Context Family Report

This report isolates a fixed-context comparison at `ctx256` to help interpret why positive-regime onset differs across families.

## Fixed `ctx256` Comparison

| Family | Attention Share | KV Bytes | KV Bytes / Token | Memory Pressure | Online | AdaptiveFeature | KVRegime | Oracle | Regime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-1.5B (4-bit) | 0.542 | 1,655,808 | 6,144 | 0.9963 | 1.000 | 1.000 | 1.000 | 1.274 | weak |
| Qwen2.5-1.5B (8-bit) | 0.574 | 1,655,808 | 6,144 | 0.9963 | 1.000 | 1.000 | 1.000 | 1.277 | weak |
| Qwen3-8B (4-bit) | 0.575 | 4,415,488 | 16,384 | 0.9963 | 1.156 | 1.000 | 1.156 | 1.277 | positive |
| SmolLM2-1.7B (4-bit) | 0.545 | 2,232,320 | 8,192 | 0.9963 | 1.146 | 1.000 | 1.146 | 1.275 | positive |
| SmolLM2-1.7B (8-bit) | 0.560 | 2,232,320 | 8,192 | 0.9963 | 1.151 | 1.000 | 1.151 | 1.276 | positive |
| Phi-3.5-mini (4-bit) | 0.517 | 3,643,392 | 13,824 | 0.9970 | 1.140 | 1.000 | 1.140 | 1.276 | positive |

## Main Readout

- At the same context length (`ctx256`), the weak `Qwen2.5` point carries substantially smaller estimated KV traffic than the early-positive `Qwen3`, `SmolLM2`, and `Phi-3.5` points.
- The memory-pressure proxy is similarly high across these families, so the more discriminative difference at this fixed context is the KV-byte scale and KV-bytes-per-token scale rather than attention labeling alone.
- This does not prove causality, but it gives a more concrete same-context explanation for family-dependent onset than a context-only narrative.
