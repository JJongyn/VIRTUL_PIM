# Follow-Up Studies

Date: 2026-04-12

## Scope

This note packages the follow-up studies requested after the initial paper readout:

1. denser context sweeps to sharpen regime transition,
2. stronger KV-proxy mechanism figures,
3. a small cost-model robustness check,
4. a controlled batch study,
5. one extra modern-model probe if resources allow.

## Completed Now

### 1. Dense context regime-transition study

Outputs:

- `artifacts/reports/DENSE_CONTEXT_REPORT.md`
- `figures/fig7_dense_context_transition.pdf`

Representative onset windows:

- `Qwen2.5-1.5B-Instruct 4bit`: online transition `256 -> 384`
- `Qwen3-8B 4bit`: online transition `128 -> 256`
- `SmolLM2-1.7B-Instruct 4bit`: online transition `128 -> 256`

Readout:

- Context now shows a much sharper regime-entry boundary than the earlier `256 / 512 / 768` coarse sweep.
- The onset is family-dependent, but context remains the dominant transition axis.
- This is stronger than saying "ctx512 helps"; it supports the cleaner statement that regime entry is context-dominated but family-shifted.

### 2. KV-proxy mechanism packaging

Outputs:

- `figures/fig8_context_kv_speedup_dualaxis.pdf`
- `figures/fig9_memory_pressure_vs_speedup.pdf`
- `artifacts/reports/KV_MECHANISM_REPORT.md`

Readout:

- The dual-axis figure makes the "context -> KV bytes -> speedup" story much easier to communicate.
- The scatter view shows that memory-pressure proxy tracks positive offload regimes across model families.
- This materially upgrades the paper from a pure observation story to a mechanism-supported story, even though the evidence remains proxy-based rather than counter-based.

### 3. Cost-model robustness study

Outputs:

- `artifacts/reports/COST_SENSITIVITY_REPORT.md`
- `artifacts/reports/COST_SENSITIVITY_SUMMARY.md`

Readout:

- Canonical weak cases remain weak across conservative / default / optimistic presets.
- Canonical long-context positive cases remain positive across all three presets.
- The only borderline exceptions are early-onset short-context cases (`Qwen3 @ ctx256`, `SmolLM2 @ ctx256`), which are still informative because they show that family-dependent onset exists but can sit near the decision boundary.

### 4. Controlled batch study

Outputs:

- `artifacts/reports/BATCH_CONTROL_REPORT.md`
- `figures/fig10_batch_context_control.pdf`

Readout:

- For `Qwen2.5-1.5B-Instruct 4bit`, batch strongly scales KV bytes at fixed context.
- But online speedup stays `1.000` across `bs=1,2,4,8` at `ctx256`, and stays positive but nearly flat (`1.186 -> 1.194`) across `bs=1,2,4,8` at `ctx768`.
- In the current setup, batch mostly modulates magnitude inside a regime, while context is what flips the regime.

### 5. One extra modern-model probe

Outputs:

- `artifacts/reports/microsoft_phi_3_5_mini_instruct_4bit_ctx256_bs1_phi35_probe_trace_summary.json`
- `artifacts/reports/microsoft_phi_3_5_mini_instruct_4bit_ctx256_bs1_phi35_probe_simulation.json`
- `artifacts/reports/microsoft_phi_3_5_mini_instruct_4bit_ctx384_bs1_phi35_probe_trace_summary.json`
- `artifacts/reports/microsoft_phi_3_5_mini_instruct_4bit_ctx384_bs1_phi35_probe_simulation.json`
- `artifacts/reports/microsoft_phi_3_5_mini_instruct_4bit_ctx768_bs1_phi35_probe_trace_summary.json`
- `artifacts/reports/microsoft_phi_3_5_mini_instruct_4bit_ctx768_bs1_phi35_probe_simulation.json`
- `artifacts/reports/PHI35_ONSET_REPORT.md`
- `artifacts/reports/PHI35_COST_SENSITIVITY_REPORT.md`
- `artifacts/reports/PHI35_COST_SENSITIVITY_SUMMARY.md`

Readout:

- We added `microsoft/Phi-3.5-mini-instruct` as an additional modern instruct family.
- The run succeeded even while `cuda:0` was partially occupied by another long-running training job, which means a small 4-bit validation probe is feasible without waiting for a completely idle GPU.
- `Phi-3.5-mini-instruct @ ctx256`:
  - attention share `0.517`
  - online `1.140x`
  - kv_regime `1.140x`
  - oracle `1.276x`
- `Phi-3.5-mini-instruct @ ctx384`:
  - attention share `0.609`
  - online `1.181x`
  - adaptive_feature `1.181x`
  - kv_regime `1.181x`
  - oracle `1.297x`
- `Phi-3.5-mini-instruct @ ctx768`:
  - attention share `0.492`
  - online `1.175x`
  - adaptive_feature `1.175x`
  - kv_regime `1.175x`
  - oracle `1.329x`

Interpretation:

- This strengthens the “not just old checkpoints” argument further.
- `Phi-3.5-mini-instruct` behaves more like the early-positive families than the delayed-onset `Qwen2.5` family.
- The added `ctx384` point shows that this is not a coarse two-point accident; the family stays positive through the mid-context regime as well.
- `Phi-3.5` robustness is also useful:
  - `ctx768` stays positive across conservative / default / optimistic settings
  - `ctx256` remains positive under all three settings, but sits much closer to the threshold under the conservative preset (`1.046x`)
- The extra model therefore adds both breadth and another data point for family-dependent onset.

## Bottom Line

The requested follow-ups materially strengthen the workshop version:

- regime-transition evidence is now sharper,
- mechanism evidence is clearer,
- robustness is better defended,
- and the "batch < context" claim now has a proper controlled experiment behind it.
