#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_attention_share(summary_payload: dict) -> float:
    for row in summary_payload.get("regions", []):
        if row["region"] == "attention":
            return row["latency_share"]
    raise KeyError("attention row not found")


def get_mlp_share(summary_payload: dict) -> float:
    for row in summary_payload.get("regions", []):
        if row["region"] == "mlp":
            return row["latency_share"]
    raise KeyError("mlp row not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a merged CPU/GPU correlation dataset from trace summaries and simulation reports.")
    parser.add_argument("--output-json", default="artifacts/reports/correlation_dataset.json")
    parser.add_argument("--output-csv", default="artifacts/reports/correlation_dataset.csv")
    args = parser.parse_args()

    entries = [
        ("cpu", "gpt2", 256, 1, "artifacts/reports/gpt2_ctx256_trace_summary.json", "artifacts/reports/gpt2_ctx256_simulation_prior_work.json"),
        ("cpu", "gpt2", 512, 1, "artifacts/reports/gpt2_ctx512_trace_summary.json", "artifacts/reports/gpt2_ctx512_simulation_prior_work.json"),
        ("cpu", "gpt2", 768, 1, "artifacts/reports/gpt2_ctx768_trace_summary.json", "artifacts/reports/gpt2_ctx768_simulation_prior_work.json"),
        ("cpu", "gpt2", 960, 1, "artifacts/reports/gpt2_ctx960_trace_summary.json", "artifacts/reports/gpt2_ctx960_simulation_prior_work.json"),
        ("cpu", "gpt2", 512, 4, "artifacts/reports/gpt2_ctx512_bs4_trace_summary.json", "artifacts/reports/gpt2_ctx512_bs4_simulation_prior_work.json"),
        ("cpu", "opt-125m", 256, 1, "artifacts/reports/facebook_opt_125m_ctx256_trace_summary.json", "artifacts/reports/opt125m_ctx256_simulation_prior_work.json"),
        ("cpu", "opt-125m", 512, 1, "artifacts/reports/facebook_opt_125m_ctx512_trace_summary.json", "artifacts/reports/opt125m_ctx512_simulation_prior_work.json"),
        ("cpu", "opt-125m", 768, 1, "artifacts/reports/facebook_opt_125m_ctx768_trace_summary.json", "artifacts/reports/opt125m_ctx768_simulation_prior_work.json"),
        ("cpu", "opt-125m", 512, 4, "artifacts/reports/facebook_opt_125m_ctx512_bs4_trace_summary.json", "artifacts/reports/opt125m_ctx512_bs4_simulation_prior_work.json"),
        ("gpu", "gpt2", 256, 1, "artifacts/reports/gpt2_gpu_ctx256_bs1_trace_summary.json", "artifacts/reports/gpt2_gpu_ctx256_bs1_simulation.json"),
        ("gpu", "gpt2", 512, 1, "artifacts/reports/gpt2_gpu_ctx512_bs1_trace_summary.json", "artifacts/reports/gpt2_gpu_ctx512_bs1_simulation.json"),
        ("gpu", "gpt2", 768, 1, "artifacts/reports/gpt2_gpu_ctx768_bs1_trace_summary.json", "artifacts/reports/gpt2_gpu_ctx768_bs1_simulation.json"),
        ("gpu", "gpt2", 960, 1, "artifacts/reports/gpt2_gpu_ctx960_bs1_trace_summary.json", "artifacts/reports/gpt2_gpu_ctx960_bs1_simulation.json"),
        ("gpu", "gpt2", 512, 2, "artifacts/reports/gpt2_gpu_ctx512_bs2_trace_summary.json", "artifacts/reports/gpt2_gpu_ctx512_bs2_simulation.json"),
        ("gpu", "opt-125m", 256, 1, "artifacts/reports/opt125m_gpu_ctx256_bs1_trace_summary.json", "artifacts/reports/opt125m_gpu_ctx256_bs1_simulation.json"),
        ("gpu", "opt-125m", 512, 1, "artifacts/reports/opt125m_gpu_ctx512_bs1_trace_summary.json", "artifacts/reports/opt125m_gpu_ctx512_bs1_simulation.json"),
        ("gpu", "opt-125m", 512, 2, "artifacts/reports/opt125m_gpu_ctx512_bs2_trace_summary.json", "artifacts/reports/opt125m_gpu_ctx512_bs2_simulation.json"),
    ]

    rows = []
    for device_kind, model, context, batch_size, summary_path, sim_path in entries:
        summary = load_json(summary_path)
        simulation = load_json(sim_path)
        results = {row["policy_name"]: row for row in simulation["results"]}
        rows.append(
            {
                "device_kind": device_kind,
                "model": model,
                "context": context,
                "batch_size": batch_size,
                "attention_share": get_attention_share(summary),
                "mlp_share": get_mlp_share(summary),
                "online_speedup": results["online_predictor"]["speedup_vs_gpu_only"],
                "adaptive_speedup": results["adaptive_score"]["speedup_vs_gpu_only"],
                "oracle_speedup": results["oracle"]["speedup_vs_gpu_only"],
                "online_gap_closed": results["online_predictor"]["oracle_gap_closed"],
                "adaptive_gap_closed": results["adaptive_score"]["oracle_gap_closed"],
            }
        )

    payload = {"num_points": len(rows), "rows": rows}
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")
    print(f"num_points={len(rows)}")


if __name__ == "__main__":
    main()
