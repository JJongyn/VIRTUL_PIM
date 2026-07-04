#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def run_one(
    *,
    env_name: str,
    model: str,
    quantization: str,
    context: int,
    batch_size: int,
    max_new_tokens: int,
    tag: str,
    use_ssd_cache: bool,
) -> dict:
    cmd = [
        "conda", "run", "-n", env_name, "python", "scripts/run_latest_quant_probe.py",
        "--model", model,
        "--device", "cuda:0",
        "--prompt-tokens", str(context),
        "--batch-size", str(batch_size),
        "--max-new-tokens", str(max_new_tokens),
        "--quantization", quantization,
        "--tag", tag,
    ]
    env = None
    if use_ssd_cache:
        env = {
            "HF_HOME": str(ROOT / ".hf-cache"),
            "TRANSFORMERS_CACHE": str(ROOT / ".hf-cache"),
        }
    subprocess.run(cmd, check=True, cwd=ROOT, env=None if env is None else {**os.environ, **env})

    base = f"{slugify(model)}_{quantization}_ctx{context}_bs{batch_size}_{tag}"
    summary_path = ROOT / "artifacts" / "reports" / f"{base}_trace_summary.json"
    sim_path = ROOT / "artifacts" / "reports" / f"{base}_simulation.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sim = json.loads(sim_path.read_text(encoding="utf-8"))
    sim_rows = {row["policy_name"]: row for row in sim["results"]}
    return {
        "context": context,
        "batch_size": batch_size,
        "attention_share": next(r["latency_share"] for r in summary["regions"] if r["region"] == "attention"),
        "mean_kv_bytes_est": summary.get("mean_kv_bytes_est", 0.0),
        "mean_memory_pressure_proxy": summary.get("mean_memory_pressure_proxy", 0.0),
        "online_speedup": sim_rows["online_predictor"]["speedup_vs_gpu_only"],
        "adaptive_feature_speedup": sim_rows["adaptive_feature"]["speedup_vs_gpu_only"],
        "kv_regime_speedup": sim_rows["kv_regime"]["speedup_vs_gpu_only"],
        "oracle_speedup": sim_rows["oracle"]["speedup_vs_gpu_only"],
        "trace_summary": str(summary_path.relative_to(ROOT)),
        "simulation_report": str(sim_path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled GPU batch study at two fixed contexts.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="4bit")
    parser.add_argument("--contexts", default="256,768")
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--env", default="jongyun_2026")
    parser.add_argument("--tag", default="batchctl")
    parser.add_argument("--use-ssd-cache", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    contexts = [int(x) for x in args.contexts.split(",") if x]
    batches = [int(x) for x in args.batches.split(",") if x]
    rows = []
    for context in contexts:
        for batch_size in batches:
            rows.append(
                run_one(
                    env_name=args.env,
                    model=args.model,
                    quantization=args.quantization,
                    context=context,
                    batch_size=batch_size,
                    max_new_tokens=args.max_new_tokens,
                    tag=args.tag,
                    use_ssd_cache=args.use_ssd_cache,
                )
            )

    payload = {
        "model": args.model,
        "quantization": args.quantization,
        "contexts": contexts,
        "batches": batches,
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
