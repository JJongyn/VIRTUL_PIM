# PAPI Virtual Scheduler

Software-only framework for studying **virtual PIM scheduling** on real or synthetic LLM decode traces.

## Scope

This repository does **not** reproduce the PAPI hardware architecture. It provides:

- a decode trace schema,
- a lightweight profiler interface,
- a virtual PIM cost model,
- dispatch policies,
- a trace-driven simulator.

## Layout

- `src/papi_virtual/schema.py`: trace and simulation data structures
- `src/papi_virtual/profiler.py`: trace IO and starter profiler utilities
- `src/papi_virtual/cost_model.py`: transparent virtual PIM latency model
- `src/papi_virtual/policies.py`: GPU-only, static, heuristic, online, oracle policies
- `src/papi_virtual/simulator.py`: policy evaluation loop
- `scripts/profile_decode.py`: writes a starter synthetic trace
- `scripts/run_trace_simulation.py`: evaluates all default policies on a trace

## Quick Start

Generate a starter trace:

```bash
python3 scripts/profile_decode.py --output artifacts/traces/synthetic_decode.jsonl
```

Run policy comparison:

```bash
python3 scripts/run_trace_simulation.py \
  --trace artifacts/traces/synthetic_decode.jsonl \
  --output artifacts/reports/simulation_report.json
```

## Next Integration Step

Replace the synthetic trace writer with real instrumentation from:

- `Transformers` generation loop hooks, or
- `vLLM` decode-step instrumentation.

The integration target is to emit one `KernelRecord` per decode region and step.

## Pre-Run Workflow

Before launching real experiments, the repository now supports a full dry-run preparation flow:

1. Generate a sweep plan:

```bash
python3 scripts/generate_sweep_plan.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output artifacts/plans/sweep_plan.json
```

2. If `torch` and `transformers` are installed, collect a real trace:

```bash
python3 scripts/trace_transformers_decode.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --prompt "Explain processing-in-memory for LLM decoding." \
  --max-new-tokens 32 \
  --output artifacts/traces/real_decode.jsonl
```

3. Summarize the trace:

```bash
python3 scripts/analyze_trace.py \
  --trace artifacts/traces/real_decode.jsonl
```

4. Run policy comparison:

```bash
python3 scripts/run_trace_simulation.py \
  --trace artifacts/traces/real_decode.jsonl \
  --output artifacts/reports/real_simulation.json
```

## Important Limitation

The current Transformers adapter is a **coarse step-level adapter**. It uses model metadata and step latency to derive attention/MLP region records. This is the right first implementation cut for stability, but it is not yet a kernel-level profiler.

## Smallest Practical Setup

For the lowest VRAM footprint, use:

- environment: `jongyun_2026`
- device: `cpu`
- demo model: `sshleifer/tiny-gpt2`

One-command smoke run:

```bash
conda run -n jongyun_2026 python scripts/run_minimal_pipeline.py
```

If you want a slightly more meaningful but still small model after that, use `distilgpt2`.
