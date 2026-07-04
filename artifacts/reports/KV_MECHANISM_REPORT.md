# KV Mechanism Report

- Focused GPU points: 13

## Correlations

- context vs mean_kv_bytes_est: 0.509
- context vs online_speedup: 0.691
- mean_kv_bytes_est vs online_speedup: 0.585
- mean_kv_bytes_est vs kv_regime_speedup: 0.638
- mean_kv_bytes_per_token_est vs online_speedup: 0.402
- memory_pressure_proxy vs online_speedup: 0.733
- memory_pressure_proxy vs kv_regime_speedup: 0.573
- mean_kv_bytes_est vs oracle_speedup: 0.607
- memory_pressure_proxy vs oracle_speedup: 0.935

## Main Readout

- This report tests the mechanism story directly on refreshed GPU traces with explicit KV proxies.
- If context tracks KV bytes strongly and KV bytes track speedup positively, the long-context story is better grounded than a context-only correlation.
- Memory-pressure proxy is expected to be model-family dependent; the same context can yield different positive-regime onsets.
- The `kv_regime` policy uses only KV-related runtime features and avoids direct cost-model internals.
- `kv_regime` beats `adaptive_feature` on 2/13 focused points and ties on 10/13.
- At matched context 512, mean batch-1 online speedup is 1.164 vs 1.166 for batch-2, which supports the earlier observation that batch is a weaker driver than context in the current setup.

## Family-Matched Onset

- `gpt2`: first online-positive context = 512, first adaptive-feature-positive context = 512, first kv-regime-positive context = 768
- `qwen2.5-1.5b-instruct-4bit`: first online-positive context = 512, first adaptive-feature-positive context = 512, first kv-regime-positive context = 512
- `qwen3-8b-4bit`: first online-positive context = 256, first adaptive-feature-positive context = 512, first kv-regime-positive context = 256
- `smollm2-1.7b-instruct-4bit`: first online-positive context = 256, first adaptive-feature-positive context = 512, first kv-regime-positive context = 256

## Points

| Model | Context | Batch | Attn Share | KV Bytes | KV Bytes/Token | Mem Pressure | Online | AdaptiveFeature | KVRegime | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2 | 256 | 1 | 0.461 | 837120.0 | 3072.000 | 0.996 | 1.000 | 1.000 | 1.000 | 1.269 |
| gpt2 | 512 | 1 | 0.462 | 1648128.0 | 3072.000 | 0.998 | 1.142 | 1.141 | 1.000 | 1.297 |
| gpt2 | 768 | 1 | 0.455 | 2459136.0 | 3072.000 | 0.999 | 1.160 | 1.160 | 1.160 | 1.321 |
| qwen2.5-1.5b-instruct-4bit | 256 | 1 | 0.542 | 1655808.0 | 6144.000 | 0.996 | 1.000 | 1.000 | 1.000 | 1.275 |
| qwen2.5-1.5b-instruct-4bit | 512 | 1 | 0.539 | 3259392.0 | 6144.000 | 0.998 | 1.170 | 1.170 | 1.170 | 1.306 |
| qwen2.5-1.5b-instruct-4bit | 768 | 1 | 0.539 | 4862976.0 | 6144.000 | 0.999 | 1.194 | 1.194 | 1.194 | 1.337 |
| qwen3-8b-4bit | 256 | 1 | 0.575 | 4415488.0 | 16384.000 | 0.996 | 1.156 | 1.000 | 1.156 | 1.277 |
| qwen3-8b-4bit | 512 | 1 | 0.553 | 8691712.0 | 16384.000 | 0.998 | 1.175 | 1.175 | 1.175 | 1.308 |
| qwen3-8b-4bit | 768 | 1 | 0.540 | 12967936.0 | 16384.000 | 0.999 | 1.195 | 1.195 | 1.195 | 1.337 |
| smollm2-1.7b-instruct-4bit | 256 | 1 | 0.545 | 2232320.0 | 8192.000 | 0.996 | 1.146 | 1.000 | 1.146 | 1.275 |
| smollm2-1.7b-instruct-4bit | 512 | 1 | 0.540 | 4395008.0 | 8192.000 | 0.998 | 1.171 | 1.171 | 1.171 | 1.307 |
| smollm2-1.7b-instruct-4bit | 512 | 2 | 0.528 | 8790016.0 | 16384.000 | 0.998 | 1.166 | 1.166 | 1.166 | 1.306 |
| smollm2-1.7b-instruct-4bit | 768 | 1 | 0.566 | 6557696.0 | 8192.000 | 0.999 | 1.207 | 1.207 | 1.207 | 1.343 |
