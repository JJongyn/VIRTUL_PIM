# Policy Ideas

Date: 2026-04-12

## Goal

Find a stronger but still realistic online policy for the trace-driven virtual PIM framework.

Current baseline situation:

- `adaptive_score`: too oracle-like
- `adaptive_feature`: realistic but conservative
- `adaptive_family`: captures one useful family-dependent onset case, but does not beat `adaptive_feature` on average

This means the next policy should focus on **detecting the onset of a positive offload regime** rather than simply being more aggressive everywhere.

## Ranked Ideas

### 1. Regime-Onset Policy

- **Summary**: Detect whether the current decode step has entered a memory-pressure regime, and only then allow PIM offload.
- **Core hypothesis**: The main decision problem is not “which region is attention?” but “has this request entered the long-context / KV-cache-heavy regime yet?”
- **Minimum experiment**:
  - Add features such as:
    - `context_len`
    - `estimated_kv_bytes`
    - `attention_share_proxy`
    - `bytes_per_token`
  - Build a small score that first predicts `weak-signal` vs `positive-signal`.
  - If positive-signal, use current `adaptive_feature` dispatch logic; otherwise keep everything on GPU.
- **Why it is stronger than `adaptive_feature`**:
  - `adaptive_feature` currently applies one score everywhere.
  - This explicitly models the empirical structure already seen in the results.
- **Main risk**:
  - Reviewer may say this is just a reframing of thresholding.
  - Must show that regime detection is more stable than a fixed context threshold.
- **Estimated effort**: `4-8 hours`

### 2. KV-Aware Policy

- **Summary**: Dispatch based on an explicit KV-cache traffic proxy instead of generic attention bytes alone.
- **Core hypothesis**: Offload value is tied more directly to KV-cache read pressure than to raw attention share.
- **Minimum experiment**:
  - Extend trace records with:
    - `kv_bytes_est`
    - `q_bytes_est`
    - `attn_out_bytes_est`
  - Use `kv_bytes_est / total_bytes` and `kv_bytes_est / token` as policy features.
- **Why it is stronger than `adaptive_feature`**:
  - Better aligned with the paper story: `context -> KV traffic -> offload value`.
  - More hardware-meaningful than a generic region bias.
- **Main risk**:
  - KV bytes are still proxies, not counters.
  - Need to explain the estimation formula clearly.
- **Estimated effort**: `0.5-1 day`

### 3. Hysteresis Policy

- **Summary**: Once a request enters the positive regime, keep using PIM for several future steps unless the signal clearly drops.
- **Core hypothesis**: Offload value in decode is temporally sticky; step-to-step decisions should not flip too eagerly.
- **Minimum experiment**:
  - Add a per-sequence state:
    - `entered_positive_regime`
    - `cooldown_steps`
  - Dispatch after onset with hysteresis instead of independent per-record scoring.
- **Why it is stronger than `adaptive_feature`**:
  - More realistic online scheduler behavior.
  - Can reduce under-dispatch near onset and avoid noisy threshold misses.
- **Main risk**:
  - Current simulator is largely stateless; stateful policy may look more complex than necessary.
- **Estimated effort**: `0.5 day`

### 4. Two-Stage Policy

- **Summary**: First decide whether the request is in a PIM-favorable regime, then decide which regions to offload.
- **Core hypothesis**: The current one-stage policy mixes regime recognition and region selection into a single score, which is suboptimal.
- **Minimum experiment**:
  - Stage 1: request-level regime classifier using context, KV proxy, batch, model family
  - Stage 2: region-level dispatch using attention/bytes/intensity
- **Why it is stronger than `adaptive_feature`**:
  - Cleaner decision structure
  - Easier to explain in the paper
  - Naturally handles family-dependent onset
- **Main risk**:
  - Slightly more moving parts
  - Needs careful wording to avoid sounding over-engineered
- **Estimated effort**: `1 day`

### 5. Family-Calibrated Threshold Policy

- **Summary**: Keep the policy simple, but calibrate the onset threshold per model family or architecture trait.
- **Core hypothesis**: The main missing variable is not a richer score but a family-specific threshold.
- **Minimum experiment**:
  - Replace one global `min_context_len` with per-family thresholds:
    - `gpt2`
    - `opt`
    - `qwen`
    - `smollm`
  - Keep all other features simple.
- **Why it is stronger than `adaptive_feature`**:
  - Still simple
  - Directly addresses the `SmolLM2` / `Qwen3` onset mismatch
- **Main risk**:
  - Reviewer may see it as hand-tuning
  - Must justify thresholds using held-out traces or architecture traits
- **Estimated effort**: `3-5 hours`

### 6. Tiny Learned Policy

- **Summary**: Fit a logistic regression or shallow tree on trace features to imitate oracle decisions.
- **Core hypothesis**: A tiny learned classifier can capture nonlinear onset behavior better than handwritten scores while staying interpretable.
- **Minimum experiment**:
  - Features:
    - context
    - batch
    - bytes moved
    - intensity
    - latency
    - model family
    - KV proxy once added
  - Label: oracle GPU/PIM decision
  - Evaluate on held-out model families
- **Why it is stronger than `adaptive_feature`**:
  - Still cheap and interpretable
  - Easier to optimize than hand-tuned scores
- **Main risk**:
  - Can look like overfitting if not validated carefully
  - Must keep the model tiny
- **Estimated effort**: `1 day`

### 7. Confidence-Gated Policy

- **Summary**: Only offload when the policy score is sufficiently above a confidence margin.
- **Core hypothesis**: The realistic policy loses because it is either too conservative or too noisy near the boundary; confidence gating can reduce bad dispatches.
- **Minimum experiment**:
  - Keep current feature score
  - Add an uncertainty margin around the threshold
  - Only dispatch if above `threshold + margin`
- **Why it is stronger than `adaptive_feature`**:
  - Safer than simply lowering the threshold
  - Better control of false-positive offloads
- **Main risk**:
  - May simply become “more conservative `adaptive_feature`”
- **Estimated effort**: `2-4 hours`

### 8. Delta-Based Policy

- **Summary**: Use the change in context / bytes / attention share across recent steps instead of only the absolute values.
- **Core hypothesis**: Entering the positive regime is a transition phenomenon, so deltas can detect it earlier.
- **Minimum experiment**:
  - Add:
    - `delta_context`
    - `delta_kv_bytes`
    - `delta_attention_proxy`
  - Trigger offload when growth rate crosses a threshold.
- **Why it is stronger than `adaptive_feature`**:
  - Better at detecting onset rather than just current state
- **Main risk**:
  - Current trace format may need stateful aggregation
  - Gains may be small if context grows linearly in all cases
- **Estimated effort**: `0.5-1 day`

## Best Fit For This Project

### Best immediate idea: Regime-Onset Policy

This is the best next step because:

- It matches the strongest empirical result already observed.
- It is easy to explain.
- It stays realistic.
- It strengthens the paper story directly.

### Best medium-term idea: KV-Aware Policy

This is the best idea if the goal is to make the paper stronger, because it ties the policy to the KV-cache bottleneck story.

### Best method-style idea: Tiny Learned Policy

This is the strongest policy-method contribution if you want to push beyond characterization, but it is riskier and easier to attack for overfitting.

## Recommended Execution Order

1. Implement `Regime-Onset Policy`
2. Add KV-cache proxy fields
3. Upgrade to `KV-Aware Regime-Onset Policy`
4. Only then consider a tiny learned policy

## Bottom Line

The next good policy should not try to be globally more aggressive.

It should answer:

**“Has this decode stream entered a KV-cache-driven positive offload regime yet?”**

That is the cleanest policy idea supported by the current experiments.
