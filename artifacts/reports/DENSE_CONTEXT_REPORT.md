# Dense Context Report

This report sharpens regime-transition boundaries with denser context sweeps.

## Qwen/Qwen2.5-1.5B-Instruct

- first online-positive context: 384
- last online-weak context: 256
- online transition window: 256 -> 384
- first adaptive-feature-positive context: 384
- first kv-regime-positive context: 512
- last kv-regime-weak context: 384
- kv-regime transition window: 384 -> 512

| Context | Attn Share | KV Bytes | Mem Pressure | Online | AdaptiveFeature | KVRegime | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.541 | 943104.0 | 0.994 | 1.000 | 1.000 | 1.000 | 1.260 |
| 256 | 0.537 | 1655808.0 | 0.996 | 1.000 | 1.000 | 1.000 | 1.274 |
| 384 | 0.540 | 2546688.0 | 0.998 | 1.159 | 1.159 | 1.000 | 1.292 |
| 512 | 0.537 | 3259392.0 | 0.998 | 1.169 | 1.169 | 1.169 | 1.306 |
| 640 | 0.535 | 4150272.0 | 0.999 | 1.182 | 1.182 | 1.182 | 1.323 |
| 768 | 0.541 | 4862976.0 | 0.999 | 1.195 | 1.195 | 1.195 | 1.337 |
| 1024 | 0.540 | 6466560.0 | 0.999 | 1.215 | 1.215 | 1.215 | 1.363 |

## unsloth/Qwen3-8B-unsloth-bnb-4bit

- first online-positive context: 256
- last online-weak context: 128
- online transition window: 128 -> 256
- first adaptive-feature-positive context: 384
- first kv-regime-positive context: 256
- last kv-regime-weak context: 128
- kv-regime transition window: 128 -> 256

| Context | Attn Share | KV Bytes | Mem Pressure | Online | AdaptiveFeature | KVRegime | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.588 | 2514944.0 | 0.994 | 1.000 | 1.000 | 1.000 | 1.262 |
| 256 | 0.574 | 4415488.0 | 0.996 | 1.155 | 1.000 | 1.155 | 1.277 |
| 384 | 0.563 | 6791168.0 | 0.998 | 1.167 | 1.167 | 1.167 | 1.295 |
| 512 | 0.552 | 8691712.0 | 0.998 | 1.175 | 1.175 | 1.175 | 1.308 |
| 640 | 0.511 | 11067392.0 | 0.999 | 1.172 | 1.172 | 1.172 | 1.319 |
| 768 | 0.538 | 12967936.0 | 0.999 | 1.194 | 1.194 | 1.194 | 1.336 |
| 1024 | 0.522 | 17244160.0 | 0.999 | 1.207 | 1.207 | 1.207 | 1.359 |

## HuggingFaceTB/SmolLM2-1.7B-Instruct

- first online-positive context: 256
- last online-weak context: 128
- online transition window: 128 -> 256
- first adaptive-feature-positive context: 384
- first kv-regime-positive context: 256
- last kv-regime-weak context: 128
- kv-regime transition window: 128 -> 256

| Context | Attn Share | KV Bytes | Mem Pressure | Online | AdaptiveFeature | KVRegime | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.541 | 1150976.0 | 0.993 | 1.000 | 1.000 | 1.000 | 1.258 |
| 256 | 0.558 | 2232320.0 | 0.996 | 1.150 | 1.000 | 1.150 | 1.276 |
| 384 | 0.540 | 3313664.0 | 0.998 | 1.158 | 1.158 | 1.158 | 1.291 |
| 512 | 0.539 | 4395008.0 | 0.998 | 1.170 | 1.170 | 1.170 | 1.307 |
| 640 | 0.545 | 5476352.0 | 0.999 | 1.185 | 1.185 | 1.185 | 1.323 |
| 768 | 0.547 | 6557696.0 | 0.999 | 1.198 | 1.198 | 1.198 | 1.339 |
| 1024 | 0.553 | 8450048.0 | 0.999 | 1.222 | 1.222 | 1.222 | 1.366 |
