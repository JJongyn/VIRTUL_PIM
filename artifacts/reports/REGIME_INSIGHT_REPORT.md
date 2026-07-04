# Regime Insight Report

- Total points: 27
- Online-positive points (`online_speedup >= 1.05`): 22
- Online-weak points (`online_speedup < 1.05`): 5
- Adaptive-feature-positive points: 20

## Positive vs Weak Regimes

- Mean context in online-positive regime: 599.3
- Mean context in online-weak regime: 256.0
- Mean attention share in online-positive regime: 0.545
- Mean attention share in online-weak regime: 0.588
- Mean oracle speedup in online-positive regime: 1.316
- Mean oracle speedup in online-weak regime: 1.278

## Model-Family Readout

### gpt2

- points: 10
- mean online speedup: 1.122
- mean attention share: 0.452
- first positive context: 512.0

### opt-125m

- points: 7
- mean online speedup: 1.175
- mean attention share: 0.722
- first positive context: 512.0

### qwen2.5-1.5b-instruct-4bit

- points: 3
- mean online speedup: 1.125
- mean attention share: 0.551
- first positive context: 512.0

### qwen3-8b-4bit

- points: 3
- mean online speedup: 1.156
- mean attention share: 0.506
- first positive context: 256.0

### smollm2-1.7b-instruct-4bit

- points: 4
- mean online speedup: 1.172
- mean attention share: 0.543
- first positive context: 256.0

## Main Takeaways

- Virtual PIM value is not universal; it appears in a subset of decode regimes.
- Context remains the clearest empirical separator between weak and positive regimes.
- Attention share is directionally useful but does not fully explain the regime boundary on its own.
- Model family changes the onset of positive regimes, especially on newer quantized models.

## Positive-Regime Points

| Device | Model | Context | Batch | Attention Share | Online | AdaptiveFeature | Oracle |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | gpt2 | 512 | 1 | 0.453 | 1.139 | 1.139 | 1.296 |
| gpu | gpt2 | 512 | 1 | 0.466 | 1.143 | 1.140 | 1.296 |
| gpu | gpt2 | 512 | 2 | 0.459 | 1.141 | 1.141 | 1.296 |
| cpu | gpt2 | 512 | 4 | 0.445 | 1.136 | 1.136 | 1.295 |
| cpu | gpt2 | 768 | 1 | 0.449 | 1.157 | 1.157 | 1.320 |
| gpu | gpt2 | 768 | 1 | 0.462 | 1.162 | 1.162 | 1.322 |
| cpu | gpt2 | 960 | 1 | 0.434 | 1.164 | 1.164 | 1.335 |
| gpu | gpt2 | 960 | 1 | 0.464 | 1.177 | 1.177 | 1.341 |
| cpu | opt-125m | 512 | 1 | 0.667 | 1.219 | 1.219 | 1.323 |
| gpu | opt-125m | 512 | 1 | 0.852 | 1.298 | 1.296 | 1.348 |
| gpu | opt-125m | 512 | 2 | 0.857 | 1.300 | 1.300 | 1.348 |
| cpu | opt-125m | 512 | 4 | 0.540 | 1.170 | 1.170 | 1.306 |
| cpu | opt-125m | 768 | 1 | 0.632 | 1.235 | 1.235 | 1.355 |
| gpu | qwen2.5-1.5b-instruct-4bit | 512 | 1 | 0.572 | 1.182 | 1.182 | 1.310 |
| gpu | qwen2.5-1.5b-instruct-4bit | 768 | 1 | 0.539 | 1.194 | 1.194 | 1.337 |
| gpu | qwen3-8b-4bit | 256 | 1 | 0.547 | 1.147 | 1.000 | 1.275 |
| gpu | qwen3-8b-4bit | 512 | 1 | 0.496 | 1.154 | 1.154 | 1.300 |
| gpu | qwen3-8b-4bit | 768 | 1 | 0.474 | 1.167 | 1.167 | 1.324 |
| gpu | smollm2-1.7b-instruct-4bit | 256 | 1 | 0.544 | 1.146 | 1.000 | 1.275 |
| gpu | smollm2-1.7b-instruct-4bit | 512 | 1 | 0.556 | 1.176 | 1.176 | 1.309 |
| gpu | smollm2-1.7b-instruct-4bit | 512 | 2 | 0.530 | 1.167 | 1.167 | 1.306 |
| gpu | smollm2-1.7b-instruct-4bit | 768 | 1 | 0.543 | 1.196 | 1.196 | 1.338 |
