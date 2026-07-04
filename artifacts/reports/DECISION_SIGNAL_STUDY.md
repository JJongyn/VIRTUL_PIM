# Decision Signal Study

- Workloads: 34
- Label: whether the realistic online predictor reaches a positive virtual offload regime (`speedup > 1.05x`) under the current virtual replay setup.
- This label does not represent real-hardware benefit; it is used only to compare the relative informativeness of different signal families under a fixed backend definition.
- Classifier: logistic regression with standardized features and a 0.5 decision threshold.
- Feature groups: attention-only, context-only, context+attention, KV-only, and all-features.
- Evaluation: 5 family-held-out folds and 3 context-bucket-held-out folds over the same 34 workloads.
- Label balance at 1.05x: 27/34 positive workloads.

## Family-Held-Out Summary

| Signal Family | Accuracy | F1 | ROC-AUC | Pos Recall | FP Rate | Pos Rate |
|---|---:|---:|---:|---:|---:|---:|
| attention_only | 0.685±0.197 | 0.713±0.327 | 0.662±0.275 | 0.757±0.359 | 0.500±0.500 | 0.791±0.132 |
| context_only | 0.884±0.129 | 0.928±0.079 | 0.917±0.186 | 0.875±0.134 | 0.000±0.000 | 0.791±0.132 |
| context_plus_attention | 0.828±0.138 | 0.894±0.083 | 0.917±0.186 | 0.875±0.134 | 0.167±0.373 | 0.791±0.132 |
| kv_only | 0.698±0.266 | 0.615±0.437 | 0.833±0.236 | 0.597±0.431 | 0.167±0.373 | 0.791±0.132 |
| all_features | 0.772±0.229 | 0.763±0.349 | 0.833±0.236 | 0.736±0.355 | 0.167±0.373 | 0.791±0.132 |

## Context-Bucket-Held-Out Summary

| Signal Family | Accuracy | F1 | ROC-AUC | Pos Recall | FP Rate | Pos Rate |
|---|---:|---:|---:|---:|---:|---:|
| attention_only | 0.666±0.216 | 0.777±0.174 | 0.500±0.000 | 0.878±0.092 | 0.333±0.471 | 0.788±0.300 |
| context_only | 0.640±0.267 | 0.749±0.192 | 0.500±0.000 | 0.852±0.210 | 0.333±0.471 | 0.788±0.300 |
| context_plus_attention | 0.640±0.267 | 0.749±0.192 | 0.500±0.000 | 0.852±0.210 | 0.333±0.471 | 0.788±0.300 |
| kv_only | 0.653±0.231 | 0.765±0.177 | 0.500±0.000 | 0.865±0.143 | 0.333±0.471 | 0.788±0.300 |
| all_features | 0.714±0.264 | 0.803±0.197 | 0.500±0.000 | 0.926±0.105 | 0.333±0.471 | 0.788±0.300 |

## Threshold Sensitivity

| Threshold | Family-held-out Context F1 | Family-held-out All-Features F1 | Context-held-out Context F1 | Context-held-out All-Features F1 |
|---:|---:|---:|---:|---:|
| 1.01 | 0.928 | 0.763 | 0.749 | 0.803 |
| 1.03 | 0.928 | 0.763 | 0.749 | 0.803 |
| 1.05 | 0.928 | 0.763 | 0.749 | 0.803 |
| 1.10 | 0.928 | 0.763 | 0.749 | 0.803 |

## Classifier Robustness

| Split | Metric | Logistic Context | Logistic All | Tree Context | Tree All |
|---|---|---:|---:|---:|---:|
| family-held-out | F1 | 0.928 | 0.763 | 0.928 | 0.763 |
| family-held-out | FP rate | 0.000 | 0.167 | 0.000 | 0.333 |
| context-bucket-held-out | F1 | 0.749 | 0.803 | 0.749 | 0.778 |
| context-bucket-held-out | FP rate | 0.333 | 0.333 | 0.333 | 0.333 |

## Boundary-Focused Subset

| Signal Family | F1 | FP Rate | Pos Recall |
|---|---:|---:|---:|
| attention_only | 0.573 | 0.444 | 0.625 |
| context_only | 0.698 | 0.000 | 0.625 |
| context_plus_attention | 0.665 | 0.333 | 0.625 |
| kv_only | 0.593 | 0.167 | 0.583 |
| all_features | 0.759 | 0.167 | 0.750 |

## Main Readout

- Online-positive fraction rises from 4/11 in short-context workloads to 9/9 in mid-context and 14/14 in long-context workloads.
- On family-held-out splits, `context_only` is the strongest coarse detector (F1=0.928±0.079, FP rate=0.000).
- `attention_only` over-predicts the positive regime on held-out families (positive recall=0.757, FP rate=0.500), which supports the paper's negative point that static attention labeling is too permissive.
- On context-bucket-held-out splits, `all_features` is the strongest overall detector (F1=0.803±0.197) and modestly exceeds `context_only` (F1=0.749±0.192); this supports calling KV-related signals complementary decision signals rather than a standalone dominant mechanism.
- The signal ordering is not specific to logistic regression: under a shallow decision tree, family-held-out `context_only` still exceeds `attention_only` in F1 (0.928 vs 0.655), and context-bucket-held-out `all_features` still slightly exceeds `context_only` (0.778 vs 0.749).
- On a boundary-focused subset (21 workloads with `1.00x <= online <= 1.18x`), `all_features` reduces the family-held-out false-positive rate relative to `attention_only` (0.167 vs 0.444) while remaining competitive with `context_only` in F1 (0.759 vs 0.698).
- Threshold sensitivity is limited but non-trivial: family-held-out `context_only` F1 moves from 0.928 at 1.01x to 0.928 at 1.10x, while the main qualitative ordering remains context-first and attention-too-permissive.
- The resulting interpretation is predictive rather than causal: context gives the clearest coarse regime boundary in this setup, while KV-related features help refine boundary detection when coarse context cues are held out.
