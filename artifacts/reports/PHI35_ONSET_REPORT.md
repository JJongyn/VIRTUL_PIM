# Phi-3.5 Onset Report

This report summarizes the additional modern-family validation run on `microsoft/Phi-3.5-mini-instruct` in 4-bit mode.

## Main Readout

- `Phi-3.5` enters the positive regime early and stays positive from `ctx256` through `ctx384` and `ctx768`.
- The `ctx384` point closes the coarse gap between `ctx256` and `ctx768` and shows that this is not a one-off short-context artifact.
- `kv_regime` recovers the early positive point at `ctx256`, while `adaptive_feature` still misses it there.

| Context | Attention Share | KV Bytes | Mem Pressure | Online | AdaptiveFeature | KVRegime | AdaptiveFamily | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.517 | 3643392.0 | 0.997 | 1.140 | 1.000 | 1.140 | 1.000 | 1.276 |
| 384 | 0.609 | 4970496.0 | 0.998 | 1.181 | 1.181 | 1.181 | 1.037 | 1.297 |
| 768 | 0.492 | 9836544.0 | 0.999 | 1.175 | 1.175 | 1.175 | 1.175 | 1.329 |

## Interpretation

- `Phi-3.5` behaves more like the early-positive families (`Qwen3`, `SmolLM2`) than the delayed-onset `Qwen2.5` family.
- This strengthens the claim that family-dependent onset is real and not confined to one modern model family.
