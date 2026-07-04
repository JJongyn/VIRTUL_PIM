# Quantization Variation Report

This report checks whether the main regime patterns are specific to the 4-bit setup used in the main dense-context analysis.

## Main Readout

- The delayed-onset `Qwen2.5-1.5B` pattern persists across quantization:
  - `ctx256, 4bit`: `online=1.000x`, `kv_regime=1.000x`, `oracle=1.274x`
  - `ctx256, 8bit`: `online=1.000x`, `kv_regime=1.000x`, `oracle=1.277x`
  - `ctx768, 4bit`: `online=1.195x`, `kv_regime=1.195x`, `oracle=1.337x`
  - `ctx768, 8bit`: `online=1.208x`, `kv_regime=1.208x`, `oracle=1.343x`
- The early-positive `SmolLM2-1.7B` pattern also persists across quantization:
  - `ctx256, 4bit`: `online=1.150x`, `kv_regime=1.150x`, `oracle=1.276x`
  - `ctx256, 8bit`: `online=1.151x`, `kv_regime=1.151x`, `oracle=1.276x`

## Interpretation

- For both a delayed-onset family (`Qwen2.5`) and an early-positive family (`SmolLM2`), the coarse regime assignment is unchanged between 4-bit and 8-bit runs.
- Quantization changes absolute latency and slightly shifts absolute speedup, but it does not flip the weak-versus-positive regime label on these controlled checks.
- This is still a focused robustness check, not a broad quantization study across all families and decode settings.
