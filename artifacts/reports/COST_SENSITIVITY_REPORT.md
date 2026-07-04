# Cost Sensitivity Report

This report checks whether the qualitative weak-regime / positive-regime split survives conservative, default, and optimistic virtual-PIM settings.

| Trace | Preset | Online | AdaptiveFeature | KVRegime | Oracle |
|---|---|---:|---:|---:|---:|
| gpt2_none_ctx256_bs1_kvmech.jsonl | conservative | 1.000 | 1.000 | 1.000 | 1.058 |
| gpt2_none_ctx256_bs1_kvmech.jsonl | default | 1.000 | 1.000 | 1.000 | 1.269 |
| gpt2_none_ctx256_bs1_kvmech.jsonl | optimistic | 1.000 | 1.000 | 1.000 | 1.481 |
| gpt2_none_ctx768_bs1_kvmech.jsonl | conservative | 1.080 | 1.080 | 1.080 | 1.101 |
| gpt2_none_ctx768_bs1_kvmech.jsonl | default | 1.160 | 1.160 | 1.160 | 1.321 |
| gpt2_none_ctx768_bs1_kvmech.jsonl | optimistic | 1.224 | 1.224 | 1.224 | 1.541 |
| qwen_qwen2_5_1_5b_instruct_4bit_ctx256_bs1_kvmech.jsonl | conservative | 1.000 | 1.000 | 1.000 | 1.062 |
| qwen_qwen2_5_1_5b_instruct_4bit_ctx256_bs1_kvmech.jsonl | default | 1.000 | 1.000 | 1.000 | 1.275 |
| qwen_qwen2_5_1_5b_instruct_4bit_ctx256_bs1_kvmech.jsonl | optimistic | 1.000 | 1.000 | 1.000 | 1.487 |
| qwen_qwen2_5_1_5b_instruct_4bit_ctx768_bs1_kvmech.jsonl | conservative | 1.096 | 1.096 | 1.096 | 1.114 |
| qwen_qwen2_5_1_5b_instruct_4bit_ctx768_bs1_kvmech.jsonl | default | 1.194 | 1.194 | 1.194 | 1.337 |
| qwen_qwen2_5_1_5b_instruct_4bit_ctx768_bs1_kvmech.jsonl | optimistic | 1.276 | 1.276 | 1.276 | 1.559 |
| unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx256_bs1_kvmech.jsonl | conservative | 1.049 | 1.000 | 1.049 | 1.064 |
| unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx256_bs1_kvmech.jsonl | default | 1.156 | 1.000 | 1.156 | 1.277 |
| unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx256_bs1_kvmech.jsonl | optimistic | 1.246 | 1.000 | 1.246 | 1.490 |
| unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx768_bs1_kvmech.jsonl | conservative | 1.096 | 1.096 | 1.096 | 1.114 |
| unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx768_bs1_kvmech.jsonl | default | 1.195 | 1.195 | 1.195 | 1.337 |
| unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx768_bs1_kvmech.jsonl | optimistic | 1.277 | 1.277 | 1.277 | 1.560 |
| huggingfacetb_smollm2_1_7b_instruct_4bit_ctx256_bs1_kvmech.jsonl | conservative | 1.046 | 1.000 | 1.046 | 1.063 |
| huggingfacetb_smollm2_1_7b_instruct_4bit_ctx256_bs1_kvmech.jsonl | default | 1.146 | 1.000 | 1.146 | 1.275 |
| huggingfacetb_smollm2_1_7b_instruct_4bit_ctx256_bs1_kvmech.jsonl | optimistic | 1.230 | 1.000 | 1.230 | 1.488 |
| huggingfacetb_smollm2_1_7b_instruct_4bit_ctx768_bs1_kvmech.jsonl | conservative | 1.102 | 1.102 | 1.102 | 1.119 |
| huggingfacetb_smollm2_1_7b_instruct_4bit_ctx768_bs1_kvmech.jsonl | default | 1.207 | 1.207 | 1.207 | 1.343 |
| huggingfacetb_smollm2_1_7b_instruct_4bit_ctx768_bs1_kvmech.jsonl | optimistic | 1.295 | 1.295 | 1.295 | 1.567 |
