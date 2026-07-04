#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run_one(model: str, context: int, max_new_tokens: int, min_offload_bytes: float, env_name: str) -> dict:
    tag = model.replace("/", "_").replace("-", "_")
    trace_output = f"artifacts/traces/{tag}_ctx{context}.jsonl"
    summary_json = f"artifacts/reports/{tag}_ctx{context}_trace_summary.json"
    summary_csv = f"artifacts/reports/{tag}_ctx{context}_trace_regions.csv"
    summary_md = f"artifacts/reports/{tag}_ctx{context}_trace_summary.md"
    simulation_output = f"artifacts/reports/{tag}_ctx{context}_simulation.json"

    cmd = [
        "conda", "run", "-n", env_name, "python", "scripts/run_cpu_signal_experiment.py",
        "--model", model,
        "--device", "cpu",
        "--prompt-tokens", str(context),
        "--max-new-tokens", str(max_new_tokens),
        "--min-offload-bytes", str(min_offload_bytes),
        "--trace-output", trace_output,
        "--summary-json", summary_json,
        "--summary-csv", summary_csv,
        "--summary-md", summary_md,
        "--simulation-output", simulation_output,
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)

    trace_payload = json.loads(Path(ROOT / summary_json).read_text(encoding="utf-8"))
    sim_payload = json.loads(Path(ROOT / simulation_output).read_text(encoding="utf-8"))
    sim_rows = {row["policy_name"]: row for row in sim_payload["results"]}

    return {
        "model": model,
        "context": context,
        "trace_summary": summary_json,
        "simulation": simulation_output,
        "attention_share": next(row["latency_share"] for row in trace_payload["regions"] if row["region"] == "attention"),
        "mlp_share": next(row["latency_share"] for row in trace_payload["regions"] if row["region"] == "mlp"),
        "gpu_only_speedup": sim_rows["gpu_only"]["speedup_vs_gpu_only"],
        "static_attention_speedup": sim_rows["static_attention"]["speedup_vs_gpu_only"],
        "online_predictor_speedup": sim_rows["online_predictor"]["speedup_vs_gpu_only"],
        "oracle_speedup": sim_rows["oracle"]["speedup_vs_gpu_only"],
        "online_gap_closed": sim_rows["online_predictor"]["oracle_gap_closed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CPU context sweep and aggregate the results.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--contexts", default="256,512,768,960")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-offload-bytes", type=float, default=32768.0)
    parser.add_argument("--env", default="jongyun_2026")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    contexts = [int(x) for x in args.contexts.split(",") if x]
    rows = []
    for context in contexts:
        rows.append(run_one(args.model, context, args.max_new_tokens, args.min_offload_bytes, args.env))

    payload = {
        "model": args.model,
        "contexts": contexts,
        "rows": rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    for row in rows:
        print(
            f"context={row['context']} attention_share={row['attention_share']:.3f} "
            f"online={row['online_predictor_speedup']:.3f} oracle={row['oracle_speedup']:.3f}"
        )


if __name__ == "__main__":
    main()
