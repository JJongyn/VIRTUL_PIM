# Cost Sensitivity Report

This report checks whether the qualitative weak-regime / positive-regime split survives conservative, default, and optimistic virtual-PIM settings.

| Trace | Preset | Online | AdaptiveFeature | KVRegime | Oracle |
|---|---|---:|---:|---:|---:|
| microsoft_phi_3_5_mini_instruct_4bit_ctx256_bs1_phi35_probe.jsonl | conservative | 1.046 | 1.000 | 1.046 | 1.063 |
| microsoft_phi_3_5_mini_instruct_4bit_ctx256_bs1_phi35_probe.jsonl | default | 1.140 | 1.000 | 1.140 | 1.276 |
| microsoft_phi_3_5_mini_instruct_4bit_ctx256_bs1_phi35_probe.jsonl | optimistic | 1.219 | 1.000 | 1.219 | 1.489 |
| microsoft_phi_3_5_mini_instruct_4bit_ctx768_bs1_phi35_probe.jsonl | conservative | 1.087 | 1.087 | 1.087 | 1.107 |
| microsoft_phi_3_5_mini_instruct_4bit_ctx768_bs1_phi35_probe.jsonl | default | 1.175 | 1.175 | 1.175 | 1.329 |
| microsoft_phi_3_5_mini_instruct_4bit_ctx768_bs1_phi35_probe.jsonl | optimistic | 1.247 | 1.247 | 1.247 | 1.550 |
