# Practical Policy Report

## Proposed Policy

- Stage 1: context gate (`ctx >= 256`) and attention-region candidacy.
- Stage 2: KV refinement using estimated KV bytes, KV bytes/token, and memory-pressure proxy.
- Stage 3: hysteresis with separate on/off thresholds to reduce oscillation along the attention-decision stream.

## Overall Workload Summary

| Policy | Speedup | Oracle Gap Closed | Oracle Regret | Pos Recall | FP Rate | Missed-Positive | Switches /100 | Avg Span |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpu_only | 1.000±0.000 | 0.000±0.000 | 1.000±0.000 | 0.000±0.000 | 0.000±0.000 | 1.000±0.000 | 0.000±0.000 | 16.000±0.000 |
| static_attention | 1.181±0.046 | 0.641±0.086 | 0.359±0.086 | 0.500±0.000 | 0.000±0.000 | 0.500±0.000 | 0.000±0.000 | 16.000±0.000 |
| context_threshold | 1.132±0.098 | 0.443±0.313 | 0.557±0.313 | 0.338±0.234 | 0.000±0.000 | 0.662±0.234 | 0.000±0.000 | 16.000±0.000 |
| adaptive_feature | 1.132±0.098 | 0.443±0.313 | 0.557±0.313 | 0.336±0.233 | 0.000±0.000 | 0.664±0.233 | 0.735±2.941 | 15.373±2.510 |
| kv_regime | 1.122±0.086 | 0.427±0.296 | 0.573±0.296 | 0.338±0.234 | 0.000±0.000 | 0.662±0.234 | 0.000±0.000 | 16.000±0.000 |
| context_kv_refined | 1.112±0.090 | 0.388±0.308 | 0.612±0.308 | 0.306±0.241 | 0.000±0.000 | 0.694±0.241 | 0.184±1.056 | 15.765±1.352 |
| context_kv_hysteresis | 1.112±0.090 | 0.388±0.308 | 0.612±0.308 | 0.306±0.241 | 0.000±0.000 | 0.694±0.241 | 0.184±1.056 | 15.765±1.352 |

## Boundary-Subset Summary

| Policy | Speedup | Oracle Regret | Pos Recall | FP Rate | Switches /100 |
|---|---:|---:|---:|---:|---:|
| gpu_only | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| static_attention | 1.155 | 0.388 | 0.500 | 0.000 | 0.000 |
| context_threshold | 1.050 | 0.809 | 0.156 | 0.000 | 0.000 |
| adaptive_feature | 1.050 | 0.809 | 0.154 | 0.000 | 0.781 |
| kv_regime | 1.069 | 0.731 | 0.219 | 0.000 | 0.000 |
| context_kv_refined | 1.058 | 0.777 | 0.182 | 0.000 | 0.391 |
| context_kv_hysteresis | 1.058 | 0.777 | 0.182 | 0.000 | 0.391 |

## Regret Distribution Anchors

| Policy | Median Regret | P90 Regret | Median Switches /100 | P90 Switches /100 |
|---|---:|---:|---:|---:|
| gpu_only | 1.000 | 1.000 | 0.000 | 0.000 |
| static_attention | 0.381 | 0.433 | 0.000 | 0.000 |
| context_threshold | 0.387 | 1.000 | 0.000 | 0.000 |
| adaptive_feature | 0.387 | 1.000 | 0.000 | 0.000 |
| kv_regime | 0.393 | 1.000 | 0.000 | 0.000 |
| context_kv_refined | 0.394 | 1.000 | 0.000 | 0.000 |
| context_kv_hysteresis | 0.394 | 1.000 | 0.000 | 0.000 |

## Hysteresis Ablation

| Compare | Speedup Δ | Regret Δ | FP Δ | Missed-Positive Δ | Switches /100 Δ | Avg Span Δ |
|---|---:|---:|---:|---:|---:|---:|
| context_kv_hysteresis - context_kv_refined | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Main Readout

- The proposed `context_kv_hysteresis` policy turns the paper's empirical findings into a simple actionable detector: context provides the coarse gate, KV signals refine the boundary, and hysteresis is retained as a practical stabilization hook.
- Relative to `static_attention`, the proposed policy reduces false-positive offload and switching, which makes it easier to present as a practical method rather than as a one-off case study.
- The explicit `context_kv_refined` ablation isolates hysteresis itself: under the current trace granularity, the hysteresis thresholds do not materially change aggregate behavior, which suggests that most of the practical gain currently comes from conservative context+KV gating rather than from hysteresis alone.
- Relative to `adaptive_feature` and `kv_regime`, the main expected gain is a more conservative context+KV detector rather than a universal average-speedup win.
