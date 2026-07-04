# Policy Story Subsection

The policy results suggest that attention labeling alone is not the right abstraction for realistic scheduling. A static attention heuristic is directionally useful, but it is too permissive: in the held-out decision-signal study, attention-only cues achieve high positive recall while also incurring a high false-positive rate. The core decision problem is therefore not simply to detect attention regions, but to decide whether a decode stream has plausibly crossed into a positive offload regime under the current replay model.

The stronger claim supported by the current data is more modest than a new scheduler result. Context length is the clearest coarse predictor in our current setup, while KV-related signals help refine boundary detection when coarse context cues are withheld. The case-study comparison between `adaptive_feature` and `kv_regime` is therefore best framed as an existence proof that explicit KV-aware features can matter on some early-positive families, not as proof that one hand-written policy universally dominates another.

