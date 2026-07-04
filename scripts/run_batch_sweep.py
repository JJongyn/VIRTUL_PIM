#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def run_one(model: str, context: int, batch: int, max_new_tokens: int, min_offload_bytes: float, env_name: str) -> dict:
    tag = model.replace("/", "_").replace("-", "_")
    trace_output = f"artifacts/traces/{tag}_ctx{context}_bs{batch}.jsonl"
    summary_json = f"artifacts/reports/{tag}_ctx{context}_bs{batch}_trace_summary.json"
    summary_csv = f"artifacts/reports/{tag}_ctx{context}_bs{batch}_trace_regions.csv"
    summary_md = f"artifacts/reports/{tag}_ctx{context}_bs{batch}_trace_summary.md"
    simulation_output = f"artifacts/reports/{tag}_ctx{context}_bs{batch}_simulation.json"

    cmd = [
        "conda", "run", "-n", env_name, "python", "scripts/run_cpu_signal_experiment.py",
        "--model", model,
        "--device", "cpu",
        "--prompt-tokens", str(context),
        "--batch-size", str(batch),
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
        "batch_size": batch,
        "attention_share": next(row["latency_share"] for row in trace_payload["regions"] if row["region"] == "attention"),
        "mlp_share": next(row["latency_share"] for row in trace_payload["regions"] if row["region"] == "mlp"),
        "online_predictor_speedup": sim_rows["online_predictor"]["speedup_vs_gpu_only"],
        "oracle_speedup": sim_rows["oracle"]["speedup_vs_gpu_only"],
        "online_gap_closed": sim_rows["online_predictor"]["oracle_gap_closed"],
        "static_attention_speedup": sim_rows["static_attention"]["speedup_vs_gpu_only"],
        "trace_summary": summary_json,
        "simulation": simulation_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CPU batch sweep and aggregate the results.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--batches", default="1,2,4")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-offload-bytes", type=float, default=32768.0)
    parser.add_argument("--env", default="jongyun_2026")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    batches = [int(x) for x in args.batches.split(",") if x]
    rows = []
    for batch in batches:
        rows.append(run_one(args.model, args.context, batch, args.max_new_tokens, args.min_offload_bytes, args.env))

    payload = {
        "model": args.model,
        "context": args.context,
        "batches": batches,
        "rows": rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    for row in rows:
        print(
            f"batch={row['batch_size']} attention_share={row['attention_share']:.3f} "
            f"online={row['online_predictor_speedup']:.3f} oracle={row['oracle_speedup']:.3f}"
        )


if __name__ == "__main__":
    main()
