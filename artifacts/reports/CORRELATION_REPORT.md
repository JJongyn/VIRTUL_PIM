# Correlation Report

- Points: 27
- CPU points: 9
- GPU points: 18

## Correlations

- attention_share vs online_speedup: 0.272
- attention_share vs adaptive_feature_speedup: 0.253
- attention_share vs adaptive_family_speedup: 0.323
- attention_share vs oracle_speedup: 0.321

## Notes

- Higher attention share is associated with higher dynamic-offload gain in the current dataset.
- `adaptive_feature` is the conservative realistic feature-only policy candidate.
- `adaptive_family` adds model-family-aware priors while staying feature-driven.
- `adaptive_score` remains useful as an oracle-friendly prototype but should not be treated as a deployment policy.

## Rows

| Device | Model | Context | Batch | Attn Share | Online | AdaptiveFeature | AdaptiveFamily | Oracle |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | gpt2 | 256 | 1 | 0.426 | 1.000 | 1.000 | 1.000 | 1.267 |
| cpu | gpt2 | 512 | 1 | 0.453 | 1.139 | 1.139 | 1.000 | 1.296 |
| cpu | gpt2 | 768 | 1 | 0.449 | 1.157 | 1.157 | 1.157 | 1.320 |
| cpu | gpt2 | 960 | 1 | 0.434 | 1.164 | 1.164 | 1.164 | 1.335 |
| cpu | gpt2 | 512 | 4 | 0.445 | 1.136 | 1.136 | 1.136 | 1.295 |
| cpu | opt-125m | 256 | 1 | 0.653 | 1.000 | 1.000 | 1.000 | 1.282 |
| cpu | opt-125m | 512 | 1 | 0.667 | 1.219 | 1.219 | 1.219 | 1.323 |
| cpu | opt-125m | 768 | 1 | 0.632 | 1.235 | 1.235 | 1.235 | 1.355 |
| cpu | opt-125m | 512 | 4 | 0.540 | 1.170 | 1.170 | 1.170 | 1.306 |
| gpu | gpt2 | 256 | 1 | 0.464 | 1.000 | 1.000 | 1.000 | 1.269 |
| gpu | gpt2 | 512 | 1 | 0.466 | 1.143 | 1.140 | 1.000 | 1.296 |
| gpu | gpt2 | 768 | 1 | 0.462 | 1.162 | 1.162 | 1.162 | 1.322 |
| gpu | gpt2 | 960 | 1 | 0.464 | 1.177 | 1.177 | 1.177 | 1.341 |
| gpu | gpt2 | 512 | 2 | 0.459 | 1.141 | 1.141 | 1.137 | 1.296 |
| gpu | opt-125m | 256 | 1 | 0.854 | 1.000 | 1.000 | 1.000 | 1.295 |
| gpu | opt-125m | 512 | 1 | 0.852 | 1.298 | 1.296 | 1.296 | 1.348 |
| gpu | opt-125m | 512 | 2 | 0.857 | 1.300 | 1.300 | 1.300 | 1.348 |
| gpu | qwen2.5-1.5b-instruct-4bit | 256 | 1 | 0.542 | 1.000 | 1.000 | 1.000 | 1.275 |
| gpu | qwen2.5-1.5b-instruct-4bit | 512 | 1 | 0.572 | 1.182 | 1.182 | 1.182 | 1.310 |
| gpu | qwen2.5-1.5b-instruct-4bit | 768 | 1 | 0.539 | 1.194 | 1.194 | 1.194 | 1.337 |
| gpu | qwen3-8b-4bit | 256 | 1 | 0.547 | 1.147 | 1.000 | 1.000 | 1.275 |
| gpu | qwen3-8b-4bit | 512 | 1 | 0.496 | 1.154 | 1.154 | 1.154 | 1.300 |
| gpu | qwen3-8b-4bit | 768 | 1 | 0.474 | 1.167 | 1.167 | 1.167 | 1.324 |
| gpu | smollm2-1.7b-instruct-4bit | 256 | 1 | 0.544 | 1.146 | 1.000 | 1.146 | 1.275 |
| gpu | smollm2-1.7b-instruct-4bit | 512 | 1 | 0.556 | 1.176 | 1.176 | 1.176 | 1.309 |
| gpu | smollm2-1.7b-instruct-4bit | 512 | 2 | 0.530 | 1.167 | 1.167 | 1.167 | 1.306 |
| gpu | smollm2-1.7b-instruct-4bit | 768 | 1 | 0.543 | 1.196 | 1.196 | 1.196 | 1.338 |
