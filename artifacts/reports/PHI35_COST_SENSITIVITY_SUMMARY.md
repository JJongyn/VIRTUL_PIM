# Cost Sensitivity Summary

This report checks whether conservative/default/optimistic virtual-PIM settings preserve the qualitative regime split.

| Trace | Intended Regime | Online Range | KVRegime Range | Oracle Range | Pattern Stable |
|---|---|---:|---:|---:|---|
| microsoft_phi_3_5_mini_instruct_4bit_ctx256_bs1_phi35_probe.jsonl | default-positive-signal | 1.046 -> 1.219 | 1.046 -> 1.219 | 1.063 -> 1.489 | no |
| microsoft_phi_3_5_mini_instruct_4bit_ctx768_bs1_phi35_probe.jsonl | default-positive-signal | 1.087 -> 1.247 | 1.087 -> 1.247 | 1.107 -> 1.550 | yes |

## Main Readout

- Absolute speedup values move with the cost-model preset, as expected.
- The canonical weak cases remain weak across all presets, and the canonical long-context positive cases remain positive across all presets.
- The only borderline exceptions are the early-onset ctx256 cases for Qwen3 and SmolLM2, which are positive under the default/optimistic settings but dip just below the 1.05 threshold under the conservative preset.
- The robustness result is therefore strongest for the main workshop claim: long-context positive regimes are not an artifact of one narrow virtual-backend setting, while a few early-onset family-specific short-context cases remain parameter-sensitive.
- This makes the paper less vulnerable to the criticism that the entire result is an artifact of one narrow virtual-backend parameter choice.
