# Paper Backbone

## Core Message

Virtual PIM value during LLM decoding is regime-dependent. The clearest coarse separator in the current setup is context-sensitive regime entry, and the observed transition is consistent with rising KV-cache-related pressure. The exact onset is family-dependent, which is why static attention rules and generic realistic heuristics can mis-handle some early-positive modern-model cases.

## Backbone Flow

1. Start with the negative result: short context does not automatically benefit from offload.
2. Show the context-transition figure: context governs regime entry, but family shifts the onset.
3. Show the KV-pressure figure: the regime shift is consistent with rising KV-cache-driven pressure.
4. Show the distributional negative-result figure: gains do not appear automatically.
5. Show the held-out decision-signal figure: attention-only over-predicts, context is the clearest coarse signal, and KV-related features are complementary.
6. Close with the case-study policy comparison plus robustness and batch-control summaries as defense rather than as the main message.

## Contribution Wording

- We provide a trace-driven framework for evaluating virtual PIM scheduling during LLM decoding.
- We show that offload value appears in context-sensitive positive regimes rather than uniformly across workloads.
- We show that context length is the clearest coarse predictor of regime entry in our current setup, and that onset varies across model families.
- We provide evidence that the observed boundary is consistent with KV-cache-related pressure and that KV-aware features can complement coarse context cues for decision support.

