# Final Research Readout

Date: 2026-04-13

## One-Line Thesis

We built a trace-driven virtual PIM evaluation framework for LLM decoding and found that offload value is not universal: it emerges reliably in longer-context, memory-pressure-heavy regimes, its onset is model-family dependent, and a KV-aware realistic policy explains that behavior better than generic attention-only heuristics.

## Paper Positioning

This is strongest as:

- a trace-driven empirical systems paper,
- a methodology + characterization paper,
- a virtual-PIM scheduling study for LLM decoding.

This is **not** strongest as:

- a real hardware speedup paper,
- a cycle-accurate PIM architecture paper,
- a final deployment-grade scheduler paper.

## What We Built

Framework components:

- decode trace schema,
- Transformers-based trace adapter,
- attention / MLP region timing hooks,
- trace analysis utilities,
- prior-work-grounded virtual PIM cost model,
- policy simulator,
- context sweep runner,
- batch sweep runner,
- GPU validation path.

Core code:

- `src/papi_virtual/schema.py`
- `src/papi_virtual/adapters/transformers_decode.py`
- `src/papi_virtual/analysis.py`
- `src/papi_virtual/cost_model.py`
- `src/papi_virtual/policies.py`
- `src/papi_virtual/simulator.py`

## Experimental Coverage

We now have evidence across:

- CPU and GPU traces,
- older small GPT-style families,
- newer 4-bit instruct families,
- dense context sweeps,
- KV-proxy mechanism studies,
- cost-model sensitivity studies,
- controlled batch studies,
- one extra modern-family validation beyond the initial readout.

Representative model families:

- `gpt2`
- `opt-125m`
- `pythia-160m`
- `pythia-410m`
- `Qwen2.5-1.5B-Instruct 4bit`
- `Qwen3-8B 4bit`
- `SmolLM2-1.7B-Instruct 4bit`
- `Phi-3.5-mini-instruct 4bit`

## Main Findings

### 1. Weak-signal vs positive-signal split is real

The strongest recurring pattern is:

- short context can be weak or borderline,
- larger context repeatedly yields positive offload value.

Examples:

- `gpt2 @ CPU, ctx256`: `online 1.000x`, `oracle 1.267x`
- `gpt2 @ GPU, ctx512`: `online 1.143x`, `oracle 1.296x`
- `opt-125m @ GPU, ctx512`: `online 1.298x`, `oracle 1.348x`
- `Qwen2.5-1.5B 4bit @ GPU, ctx768`: `online 1.194x`, `oracle 1.337x`

Interpretation:

- virtual PIM value is condition-dependent rather than automatic,
- the framework is not fabricating gains everywhere.

### 2. Context is the clearest regime separator

Dense sweeps made the transition much sharper than the earlier `256 / 512 / 768` points.

Representative onset windows:

- `Qwen2.5-1.5B 4bit`: `256 -> 384`
- `Qwen3-8B 4bit`: `128 -> 256`
- `SmolLM2-1.7B 4bit`: `128 -> 256`
- `Phi-3.5-mini-instruct 4bit`: already positive at `256`, confirmed again at `384`

Interpretation:

- context dominates regime entry,
- but the exact onset is family-dependent.

This is a better paper statement than just saying "`ctx512` helps."

### 3. Family-dependent onset is now well supported

The current evidence supports two broad modern-family behaviors:

- delayed-onset family:
  - `Qwen2.5-1.5B 4bit`
- earlier-onset families:
  - `Qwen3-8B 4bit`
  - `SmolLM2-1.7B 4bit`
  - `Phi-3.5-mini-instruct 4bit`

This matters because it upgrades the story from:

- "longer context helps"

to:

- "context governs regime entry, but onset shifts with model family."

### 4. KV pressure is the best mechanism story we currently have

The project is now stronger than a context-only observation paper.

Mechanism evidence:

- `context vs mean_kv_bytes_est = 0.509`
- `context vs online_speedup = 0.691`
- `mean_kv_bytes_est vs online_speedup = 0.585`
- `memory_pressure_proxy vs online_speedup = 0.733`
- `memory_pressure_proxy vs oracle_speedup = 0.935`

Interpretation:

- we now observe `context -> KV traffic`,
- and `KV traffic -> speedup`,
- not just `context -> speedup`.

This is still proxy-based, not direct hardware-counter evidence, but it is much closer to a real systems mechanism.

### 5. Batch is weaker than context in the current setup

Controlled batch study on `Qwen2.5-1.5B 4bit`:

- at `ctx256`, online speedup stays `1.000 -> 1.000` across `bs=1,2,4,8`
- at `ctx768`, online speedup stays positive but nearly flat: `1.186 -> 1.194`
- KV bytes scale strongly with batch,
- memory-pressure proxy barely changes within a fixed context.

Interpretation:

- batch matters,
- but in this setup it mostly changes magnitude within a regime,
- while context is what flips the regime.

### 6. Cost-model robustness is good enough for the main claim

Sensitivity over conservative / default / optimistic presets shows:

- canonical weak cases stay weak,
- canonical long-context positive cases stay positive,
- early-onset short-context cases are closer to the boundary.

For `Phi-3.5`:

- `ctx256`: `1.046x -> 1.219x`
- `ctx768`: `1.087x -> 1.247x`

Interpretation:

- absolute numbers move, as expected,
- but the main long-context positive-regime story is not an artifact of one narrow parameter setting.

## Policy Readout

Policies we evaluated:

- `gpu_only`
- `static_attention`
- `threshold`
- `online_predictor`
- `adaptive_feature`
- `kv_regime`
- `adaptive_family`
- `adaptive_score`
- `oracle`

### Current best reading

- `adaptive_score` is too oracle-like to be the main realistic contribution.
- `adaptive_feature` is still the safest realistic baseline.
- `kv_regime` is the most mechanism-aligned realistic direction.
- `adaptive_family` is useful as evidence for family-dependent onset, but not yet the best average policy.

### Important policy insight

The main realistic policy question is not:

- "is this region attention?"

It is closer to:

- "has this decode stream entered a KV-pressure-driven positive offload regime yet?"

That is why `kv_regime` is strategically more valuable than just making a generic heuristic more aggressive.

## Best Supporting Results By Theme

### Regime transition

Use:

- `artifacts/reports/DENSE_CONTEXT_REPORT.md`
- `figures/fig7_dense_context_transition.pdf`

### KV mechanism

Use:

- `artifacts/reports/KV_MECHANISM_REPORT.md`
- `figures/fig8_context_kv_speedup_dualaxis.pdf`
- `figures/fig9_memory_pressure_vs_speedup.pdf`

### Batch control

Use:

- `artifacts/reports/BATCH_CONTROL_REPORT.md`
- `figures/fig10_batch_context_control.pdf`

### Extra modern family

Use:

- `artifacts/reports/PHI35_ONSET_REPORT.md`
- `artifacts/reports/PHI35_COST_SENSITIVITY_REPORT.md`
- `artifacts/reports/PHI35_COST_SENSITIVITY_SUMMARY.md`

## Safest Claims We Can Make

We can safely claim:

1. Virtual PIM offload value during LLM decoding is regime-dependent, not universal.
2. Context length is the clearest empirical driver of regime entry in the current study.
3. The onset of positive offload value varies across model families.
4. Static offload rules are consistently weaker than dynamic policies.
5. The observed regime shift is consistent with rising KV-cache-driven memory pressure.
6. A KV-aware realistic policy explains early-onset modern-family cases better than a generic realistic feature policy.

## Claims We Should Avoid

We should avoid claiming:

1. Real hardware speedups.
2. Cycle-accurate PIM behavior.
3. Final deployment-ready scheduler quality.
4. Broad serving-scale generalization to very large production models.
5. Strong universal statements that every short-context case is weak.

## Strongest Narrative For Writing

Recommended framing:

We present a trace-driven framework for studying virtual PIM scheduling in LLM decoding and show that offload value emerges in context-sensitive positive regimes that are consistent with increased KV-cache-driven memory pressure. This regime entry is family-dependent, which explains why static rules and generic realistic heuristics miss some early-positive modern-model cases.

Recommended structure:

1. problem setup and contribution boundary,
2. framework and virtual-backend methodology,
3. regime characterization,
4. KV-pressure mechanism evidence,
5. realistic policy analysis,
6. limitations and future hardware validation.

## Remaining Gaps

What is still missing if we wanted a stronger venue:

- direct hardware-counter evidence,
- real PIM or stronger backend validation,
- broader large-model serving evidence,
- a cleaner final realistic scheduler contribution.

## Bottom Line

The project is already paper-worthy.

The strongest version is not "we built a better PIM machine" but:

- we built a useful trace-driven framework,
- we identified when virtual PIM offload becomes valuable,
- we showed that context-driven regime entry is real,
- we connected that regime shift to KV-related memory pressure,
- and we showed that family-dependent onset changes what a realistic scheduler needs to detect.
