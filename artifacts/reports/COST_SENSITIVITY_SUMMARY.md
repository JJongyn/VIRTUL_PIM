# Cost Sensitivity Summary

This report checks whether conservative/default/optimistic virtual-PIM settings preserve the qualitative regime split.

| Trace | Intended Regime | Online Range | KVRegime Range | Oracle Range | Pattern Stable |
|---|---|---:|---:|---:|---|
| gpt2_none_ctx256_bs1_kvmech.jsonl | default-weak-signal | 1.000 -> 1.000 | 1.000 -> 1.000 | 1.058 -> 1.481 | yes |
| gpt2_none_ctx768_bs1_kvmech.jsonl | default-positive-signal | 1.080 -> 1.224 | 1.080 -> 1.224 | 1.101 -> 1.541 | yes |
| huggingfacetb_smollm2_1_7b_instruct_4bit_ctx256_bs1_kvmech.jsonl | default-positive-signal | 1.046 -> 1.230 | 1.046 -> 1.230 | 1.063 -> 1.488 | no |
| huggingfacetb_smollm2_1_7b_instruct_4bit_ctx768_bs1_kvmech.jsonl | default-positive-signal | 1.102 -> 1.295 | 1.102 -> 1.295 | 1.119 -> 1.567 | yes |
| qwen_qwen2_5_1_5b_instruct_4bit_ctx256_bs1_kvmech.jsonl | default-weak-signal | 1.000 -> 1.000 | 1.000 -> 1.000 | 1.062 -> 1.487 | yes |
| qwen_qwen2_5_1_5b_instruct_4bit_ctx768_bs1_kvmech.jsonl | default-positive-signal | 1.096 -> 1.276 | 1.096 -> 1.276 | 1.114 -> 1.559 | yes |
| unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx256_bs1_kvmech.jsonl | default-positive-signal | 1.049 -> 1.246 | 1.049 -> 1.246 | 1.064 -> 1.490 | no |
| unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx768_bs1_kvmech.jsonl | default-positive-signal | 1.096 -> 1.277 | 1.096 -> 1.277 | 1.114 -> 1.560 | yes |

## Main Readout

- Absolute speedup values move with the cost-model preset, as expected.
- The canonical weak cases remain weak across all presets, and the canonical long-context positive cases remain positive across all presets.
- The only borderline exceptions are the early-onset ctx256 cases for Qwen3 and SmolLM2, which are positive under the default/optimistic settings but dip just below the 1.05 threshold under the conservative preset.
- The robustness result is therefore strongest for the main workshop claim: long-context positive regimes are not an artifact of one narrow virtual-backend setting, while a few early-onset family-specific short-context cases remain parameter-sensitive.
- This makes the paper less vulnerable to the criticism that the entire result is an artifact of one narrow virtual-backend parameter choice.
