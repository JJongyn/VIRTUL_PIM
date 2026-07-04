# Trace Summary

- Records: 32
- Steps: 16
- Context range: 784 .. 799
- Batch range: 8 .. 8
- Total latency (observed): 18531.168 ms
- Mean KV bytes (attention records): 38903808.0
- Mean KV bytes/token (attention records): 49152.000
- Mean memory-pressure proxy (attention records): 0.999

## Regions

| Region | Count | Latency (ms) | Share | Intensity |
|---|---:|---:|---:|---:|
| attention | 16 | 9630.880 | 0.520 | 0.998738 |
| mlp | 16 | 8900.288 | 0.480 | 7.972121 |
