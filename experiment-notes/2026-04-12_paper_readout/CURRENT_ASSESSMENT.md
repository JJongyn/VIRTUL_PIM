# Current Assessment

Date: 2026-04-12

## Bottom Line

The project is still paper-worthy, but the safest target remains:

**a trace-driven empirical systems paper on when virtual PIM offload value appears during LLM decoding**

and **not** a hardware-speedup paper.

## Updated Strengths

1. The virtual PIM story now extends beyond older small GPT-style checkpoints.
   - We now have positive evidence on:
     - `gpt2`
     - `opt-125m`
     - `pythia-160m`
     - `pythia-410m`
     - `Qwen2.5-1.5B-Instruct 4bit`
     - `Qwen3-8B 4bit`
     - `SmolLM2-1.7B-Instruct 4bit`

2. The weak-signal vs positive-signal split is stronger now.
   - `Qwen2.5-1.5B-Instruct 4bit` shows a clean
     - `ctx256 -> no dynamic gain`
     - `ctx512/768 -> positive dynamic gain`
   - This matches the original story and makes it much easier to defend.

3. The GPU story is no longer a one-off sanity check.
   - The aggregate dataset is now `27` points:
     - CPU: `9`
     - GPU: `18`

4. The project now has a more nuanced policy story.
   - `adaptive_feature` remains the safest realistic policy.
   - `adaptive_family` shows that model-family-dependent onset of positive regimes is a real phenomenon, even though the current implementation is not yet the best overall policy.

5. The project now has first mechanism-level evidence for the context story.
   - On a refreshed focused GPU suite with explicit KV proxies:
     - `context vs mean_kv_bytes_est = 0.509`
     - `mean_kv_bytes_est vs online_speedup = 0.585`
     - `memory_pressure_proxy vs online_speedup = 0.733`
   - This means the current paper can say more than just “longer context helps”; it can now connect that regime shift to increasing KV-cache traffic.

6. The realistic-policy story improved in a more meaningful direction.
   - `kv_regime` uses explicit KV-cache-related runtime features instead of oracle-like internals.
   - On the focused mechanism suite it improves over `adaptive_feature` on the two most important early-onset cases:
     - `Qwen3-8B @ ctx256`
     - `SmolLM2-1.7B @ ctx256`
   - This is more valuable than a small average gain because it matches the family-dependent onset hypothesis directly.

## Updated Weaknesses

1. The biggest weakness is still the virtual backend.
   - The backend is still a prior-work-grounded cost model, not a cycle-accurate PIM simulator.
   - This means we still cannot make direct hardware-speedup claims.

2. The correlation evidence is still supportive, not decisive.
   - Current aggregate correlations:
     - attention_share vs online_speedup: `0.272`
     - attention_share vs adaptive_feature_speedup: `0.253`
     - attention_share vs adaptive_family_speedup: `0.323`
     - attention_share vs oracle_speedup: `0.321`
   - These are positive, but not strong enough to carry the whole paper alone.

3. The KV/memory-pressure evidence is still proxy-based.
   - The new mechanism results are better than raw context-only correlations.
   - But they still come from trace-derived estimates, not direct hardware counters.

4. The realistic policy contribution is still incomplete.
   - `adaptive_score` is still too oracle-like.
   - `adaptive_feature` is realistic but conservative.
   - `kv_regime` is a better mechanism-aligned direction, but it is still a focused prototype rather than a final scheduler.

## What Improved Since The Previous Read

1. The “old small models only” criticism is now much weaker.
2. The “GPU evidence is too thin” criticism is weaker.
3. The “family-specific regime onset may exist” point is now empirically visible.

## What Still Blocks A Stronger Submission

1. No real hardware validation
2. No final realistic policy that clearly and consistently beats the current baseline family of policies
3. No direct counter-based proof for KV-cache bandwidth pressure
4. Still no truly large serving-scale model study beyond the small-to-mid open models we already tested

## Recommended Current Framing

Use this framing:

**We present a trace-driven virtual PIM evaluation framework for LLM decoding and show that offload value emerges repeatedly in longer-context and attention-heavy regimes, in a way that is consistent with rising KV-cache-driven memory pressure, while the exact onset can vary across model families.**

Avoid this framing:

- “We show real PIM hardware speedups.”
- “We propose a final deployment-ready scheduler.”

## Updated Readiness

- Workshop / poster / domestic venue: strong
- Mid-tier systems / architecture venue: plausible with good writing and figures
- Top-tier main conference: still short on validation depth

## Updated Scores

- Idea: `8/10`
- Current evidence: `7.5/10`
- Paper potential: `8.5/10`

## Most Important Next Steps

1. Center the paper on framework + characterization + KV-cache mechanism evidence.
2. If policy work continues, make it explicitly KV-aware rather than just family-aware.
3. Use the new mechanism figures in the main text:
   - context vs speedup
   - context vs KV bytes
   - KV bytes vs speedup

## 2026-04-13 Update

One additional modern-family validation on `microsoft/Phi-3.5-mini-instruct 4bit` further improved the story.

- `Phi-3.5 @ ctx256`: `online = 1.140x`, `kv_regime = 1.140x`, `oracle = 1.276x`
- `Phi-3.5 @ ctx384`: `online = 1.181x`, `adaptive_feature = 1.181x`, `oracle = 1.297x`
- `Phi-3.5 @ ctx768`: `online = 1.175x`, `adaptive_feature = 1.175x`, `oracle = 1.329x`

Interpretation:

- The project now has another modern instruct family showing early positive-regime onset.
- This reduces the risk that the early-onset story is specific only to `Qwen3` or `SmolLM2`.
- `Phi-3.5` sensitivity also supports the current framing:
  - `ctx768` remains positive across conservative/default/optimistic settings
  - `ctx256` remains positive but sits closer to the threshold under the conservative preset (`1.046x`)
