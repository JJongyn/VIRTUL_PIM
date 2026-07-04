# Figure Captions

## Figure 1: Context-Driven Regime Transition

Dense context sweeps across four modern instruct families show that virtual PIM benefit is not universal but emerges after a family-dependent context threshold. `Qwen2.5-1.5B` shows delayed onset (`256 -> 384`), while `Qwen3-8B`, `SmolLM2-1.7B`, and `Phi-3.5-mini` enter the positive regime earlier. In our current setup, context length is the clearest coarse predictor of regime entry, while model family shifts the onset boundary.

## Figure 2: KV-Related Pressure Is Consistent With Positive Offload Regimes

Across dense context sweeps and the extra `Phi-3.5-mini` validation, higher memory-pressure proxy values tend to align with positive online speedup. This figure provides predictive evidence that the observed regime boundary is consistent with rising KV-cache-related pressure, without claiming a fully causal mechanism.

## Figure 3: Static and Generic Realistic Policies Miss Early-Positive Cases

On selected delayed-onset and early-onset modern-model cases, `static_attention` is directionally useful but insufficient, while `kv_regime` sometimes recovers early-positive families that `adaptive_feature` misses at short context. This figure should be read as a policy case study rather than as a universal scheduler ranking.

## Figure 4: Negative Results as a Distribution, Not a Case Study

The left panel shows the workload-level distribution of realistic online gains, while the right panel shows positive-regime fraction versus context. Together they show that gains are not fabricated uniformly: short-context workloads frequently remain weak, while mid- and long-context workloads are much more often positive.

## Figure 5: Held-Out Decision-Signal Study

Held-out workload classification over the same 34 workloads uses backend-defined positive-regime labels from the current virtual replay setup. Under that fixed definition, attention-only signals over-predict positive regimes, while context length is the strongest coarse predictor. Adding KV-related features modestly improves context-bucket-held-out detection, which supports using them as complementary decision signals rather than as standalone causal proof.

## Figure 6: Same-Context Cross-Family Comparison

At a fixed `ctx=256`, delayed-onset `Qwen2.5` remains weak while early-positive `Qwen3`, `SmolLM2`, and `Phi-3.5` are already positive. The stronger cases also carry larger KV-byte scale at the same context. This gives a more concrete family-level readout than a context-only narrative, while still remaining predictive rather than causal evidence.

## Figure 7: Backend Validation Appendix Figure

The left panel shows preset-level positive fractions across conservative/default/optimistic backends, while the right panel shows first-positive context for representative families across the same presets. This makes the sensitivity argument concrete: preset changes shift absolute positivity rates, but delayed-onset families remain delayed and early-positive families remain early-positive.

## Figure 8: Family-Dependent Onset Summary

The first positive context varies by family, with `gpt2` and `opt-125m` entering later than `Qwen3`, `SmolLM2`, and `Phi-3.5`. This compact view summarizes the paper’s family-dependent onset claim.

## Figure 9: Cost-Model Robustness Summary

Conservative/default/optimistic virtual-PIM presets change absolute speedups but preserve the main qualitative conclusion: canonical long-context positive cases remain positive, while weak or borderline short-context cases stay near the boundary.

