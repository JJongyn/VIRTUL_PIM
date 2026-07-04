# Backend Validation Report

## Measured-Latency Coverage

- Total records: 1088
- Records with measured latency: 1088 (1.000)

| Region | Measured / Total | Fraction |
|---|---:|---:|
| attention | 544/544 | 1.000 |
| mlp | 544/544 | 1.000 |

## Preset Stability

| Preset | Online Positive Fraction | Adaptive Positive Fraction | KV Positive Fraction |
|---|---:|---:|---:|
| conservative | 0.500 | 0.500 | 0.500 |
| default | 0.750 | 0.500 | 0.750 |
| optimistic | 0.750 | 0.500 | 0.750 |

## Boundary Shift

| Preset | gpt2 | opt-125m | Qwen2.5 | Qwen3 | Phi-3.5 |
|---|---:|---:|---:|---:|---:|
| conservative | 768 | n/a | 768 | 768 | n/a |
| default | 768 | n/a | 768 | 256 | n/a |
| optimistic | 768 | n/a | 768 | 256 | n/a |

## Ranking Stability

| Preset | Matches Default Top-1 | Fraction |
|---|---:|---:|
| conservative | 8/8 | 1.000 |
| default | 8/8 | 1.000 |
| optimistic | 8/8 | 1.000 |

## Main Readout

- Measured latency coverage is effectively complete for the current workload set, so the GPU-side roofline fallback is not driving the main experiments.
- The backend presets change absolute speedups, but the positive-fraction ordering is preserved across conservative, default, and optimistic settings.
- Boundary movement is limited: delayed-onset families remain delayed and early-positive families remain early-positive across presets, even though the exact positive fraction changes.
- The policy ranking is also stable at the top level under preset changes, which makes the backend sensitivity argument more concrete than a generic `ordering preserved` statement.
- The sensitivity study should therefore be read as an ordering-stability check, not as a hardware calibration claim.
