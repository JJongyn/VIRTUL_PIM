# Batch Control Report

This report isolates batch size at two fixed contexts to show why context is the stronger regime driver in the current setup.

- model: Qwen/Qwen2.5-1.5B-Instruct
- quantization: 4bit
- contexts: 256, 768

## Context 256

- online speedup range across batch: 1.000 -> 1.000
- KV bytes range across batch: 1655808.0 -> 13246464.0
- memory-pressure proxy range across batch: 0.996 -> 0.996

| Batch | Attn Share | KV Bytes | Mem Pressure | Online | AdaptiveFeature | KVRegime | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.542 | 1655808.0 | 0.996 | 1.000 | 1.000 | 1.000 | 1.275 |
| 2 | 0.524 | 3311616.0 | 0.996 | 1.000 | 1.000 | 1.140 | 1.274 |
| 4 | 0.524 | 6623232.0 | 0.996 | 1.000 | 1.000 | 1.140 | 1.274 |
| 8 | 0.524 | 13246464.0 | 0.996 | 1.000 | 1.000 | 1.140 | 1.140 |

## Context 768

- online speedup range across batch: 1.186 -> 1.194
- KV bytes range across batch: 4862976.0 -> 38903808.0
- memory-pressure proxy range across batch: 0.999 -> 0.999

| Batch | Attn Share | KV Bytes | Mem Pressure | Online | AdaptiveFeature | KVRegime | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.539 | 4862976.0 | 0.999 | 1.194 | 1.194 | 1.194 | 1.337 |
| 2 | 0.525 | 9725952.0 | 0.999 | 1.188 | 1.188 | 1.188 | 1.334 |
| 4 | 0.534 | 19451904.0 | 0.999 | 1.192 | 1.192 | 1.192 | 1.336 |
| 8 | 0.520 | 38903808.0 | 0.999 | 1.186 | 1.186 | 1.186 | 1.186 |

## Main Readout

- At short context 256, online speedup stays at 1.000 -> 1.000 even as batch changes.
- At long context 768, online speedup stays positive across all tested batches: 1.186 -> 1.194.
- Context shifts memory-pressure proxy from 0.996 to 0.999, while batch mainly scales KV bytes within each fixed regime.
- This supports the claim that batch matters, but context is the stronger regime-separating variable in the current setup.
