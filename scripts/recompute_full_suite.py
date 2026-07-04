#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


ENTRIES = [
    ("cpu", "gpt2", 256, 1, "artifacts/traces/gpt2_ctx256.jsonl", "artifacts/reports/gpt2_ctx256_trace_summary.json", "artifacts/reports/gpt2_ctx256_simulation_full.json"),
    ("cpu", "gpt2", 512, 1, "artifacts/traces/gpt2_ctx512.jsonl", "artifacts/reports/gpt2_ctx512_trace_summary.json", "artifacts/reports/gpt2_ctx512_simulation_full.json"),
    ("cpu", "gpt2", 768, 1, "artifacts/traces/gpt2_ctx768.jsonl", "artifacts/reports/gpt2_ctx768_trace_summary.json", "artifacts/reports/gpt2_ctx768_simulation_full.json"),
    ("cpu", "gpt2", 960, 1, "artifacts/traces/gpt2_ctx960.jsonl", "artifacts/reports/gpt2_ctx960_trace_summary.json", "artifacts/reports/gpt2_ctx960_simulation_full.json"),
    ("cpu", "gpt2", 512, 4, "artifacts/traces/gpt2_ctx512_bs4.jsonl", "artifacts/reports/gpt2_ctx512_bs4_trace_summary.json", "artifacts/reports/gpt2_ctx512_bs4_simulation_full.json"),
    ("cpu", "opt-125m", 256, 1, "artifacts/traces/facebook_opt_125m_ctx256.jsonl", "artifacts/reports/facebook_opt_125m_ctx256_trace_summary.json", "artifacts/reports/opt125m_ctx256_simulation_full.json"),
    ("cpu", "opt-125m", 512, 1, "artifacts/traces/facebook_opt_125m_ctx512.jsonl", "artifacts/reports/facebook_opt_125m_ctx512_trace_summary.json", "artifacts/reports/opt125m_ctx512_simulation_full.json"),
    ("cpu", "opt-125m", 768, 1, "artifacts/traces/facebook_opt_125m_ctx768.jsonl", "artifacts/reports/facebook_opt_125m_ctx768_trace_summary.json", "artifacts/reports/opt125m_ctx768_simulation_full.json"),
    ("cpu", "opt-125m", 512, 4, "artifacts/traces/facebook_opt_125m_ctx512_bs4.jsonl", "artifacts/reports/facebook_opt_125m_ctx512_bs4_trace_summary.json", "artifacts/reports/opt125m_ctx512_bs4_simulation_full.json"),
    ("gpu", "gpt2", 256, 1, "artifacts/traces/gpt2_gpu_ctx256_bs1.jsonl", "artifacts/reports/gpt2_gpu_ctx256_bs1_trace_summary.json", "artifacts/reports/gpt2_gpu_ctx256_bs1_simulation_full.json"),
    ("gpu", "gpt2", 512, 1, "artifacts/traces/gpt2_gpu_ctx512_bs1.jsonl", "artifacts/reports/gpt2_gpu_ctx512_bs1_trace_summary.json", "artifacts/reports/gpt2_gpu_ctx512_bs1_simulation_full.json"),
    ("gpu", "gpt2", 768, 1, "artifacts/traces/gpt2_gpu_ctx768_bs1.jsonl", "artifacts/reports/gpt2_gpu_ctx768_bs1_trace_summary.json", "artifacts/reports/gpt2_gpu_ctx768_bs1_simulation_full.json"),
    ("gpu", "gpt2", 960, 1, "artifacts/traces/gpt2_gpu_ctx960_bs1.jsonl", "artifacts/reports/gpt2_gpu_ctx960_bs1_trace_summary.json", "artifacts/reports/gpt2_gpu_ctx960_bs1_simulation_full.json"),
    ("gpu", "gpt2", 512, 2, "artifacts/traces/gpt2_gpu_ctx512_bs2.jsonl", "artifacts/reports/gpt2_gpu_ctx512_bs2_trace_summary.json", "artifacts/reports/gpt2_gpu_ctx512_bs2_simulation_full.json"),
    ("gpu", "opt-125m", 256, 1, "artifacts/traces/opt125m_gpu_ctx256_bs1.jsonl", "artifacts/reports/opt125m_gpu_ctx256_bs1_trace_summary.json", "artifacts/reports/opt125m_gpu_ctx256_bs1_simulation_full.json"),
    ("gpu", "opt-125m", 512, 1, "artifacts/traces/opt125m_gpu_ctx512_bs1.jsonl", "artifacts/reports/opt125m_gpu_ctx512_bs1_trace_summary.json", "artifacts/reports/opt125m_gpu_ctx512_bs1_simulation_full.json"),
    ("gpu", "opt-125m", 512, 2, "artifacts/traces/opt125m_gpu_ctx512_bs2.jsonl", "artifacts/reports/opt125m_gpu_ctx512_bs2_trace_summary.json", "artifacts/reports/opt125m_gpu_ctx512_bs2_simulation_full.json"),
    ("gpu", "qwen2.5-1.5b-instruct-4bit", 256, 1, "artifacts/traces/qwen_qwen2_5_1_5b_instruct_4bit_ctx256_bs1_latest_probe.jsonl", "artifacts/reports/qwen_qwen2_5_1_5b_instruct_4bit_ctx256_bs1_latest_probe_trace_summary.json", "artifacts/reports/qwen_qwen2_5_1_5b_instruct_4bit_ctx256_bs1_latest_probe_simulation_full.json"),
    ("gpu", "qwen2.5-1.5b-instruct-4bit", 512, 1, "artifacts/traces/qwen_qwen2_5_1_5b_instruct_4bit_ctx512_bs1_latest_probe.jsonl", "artifacts/reports/qwen_qwen2_5_1_5b_instruct_4bit_ctx512_bs1_latest_probe_trace_summary.json", "artifacts/reports/qwen_qwen2_5_1_5b_instruct_4bit_ctx512_bs1_latest_probe_simulation_full.json"),
    ("gpu", "qwen2.5-1.5b-instruct-4bit", 768, 1, "artifacts/traces/qwen_qwen2_5_1_5b_instruct_4bit_ctx768_bs1_latest_probe.jsonl", "artifacts/reports/qwen_qwen2_5_1_5b_instruct_4bit_ctx768_bs1_latest_probe_trace_summary.json", "artifacts/reports/qwen_qwen2_5_1_5b_instruct_4bit_ctx768_bs1_latest_probe_simulation_full.json"),
    ("gpu", "qwen3-8b-4bit", 256, 1, "artifacts/traces/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx256_bs1_qwen3_unsloth_probe.jsonl", "artifacts/reports/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx256_bs1_qwen3_unsloth_probe_trace_summary.json", "artifacts/reports/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx256_bs1_qwen3_unsloth_probe_simulation_full.json"),
    ("gpu", "qwen3-8b-4bit", 512, 1, "artifacts/traces/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx512_bs1_qwen3_unsloth_probe.jsonl", "artifacts/reports/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx512_bs1_qwen3_unsloth_probe_trace_summary.json", "artifacts/reports/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx512_bs1_qwen3_unsloth_probe_simulation_full.json"),
    ("gpu", "qwen3-8b-4bit", 768, 1, "artifacts/traces/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx768_bs1_qwen3_unsloth_probe.jsonl", "artifacts/reports/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx768_bs1_qwen3_unsloth_probe_trace_summary.json", "artifacts/reports/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx768_bs1_qwen3_unsloth_probe_simulation_full.json"),
    ("gpu", "smollm2-1.7b-instruct-4bit", 256, 1, "artifacts/traces/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx256_bs1_latest_probe.jsonl", "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx256_bs1_latest_probe_trace_summary.json", "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx256_bs1_latest_probe_simulation_full.json"),
    ("gpu", "smollm2-1.7b-instruct-4bit", 512, 1, "artifacts/traces/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx512_bs1_latest_probe.jsonl", "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx512_bs1_latest_probe_trace_summary.json", "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx512_bs1_latest_probe_simulation_full.json"),
    ("gpu", "smollm2-1.7b-instruct-4bit", 512, 2, "artifacts/traces/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx512_bs2_latest_probe.jsonl", "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx512_bs2_latest_probe_trace_summary.json", "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx512_bs2_latest_probe_simulation_full.json"),
    ("gpu", "smollm2-1.7b-instruct-4bit", 768, 1, "artifacts/traces/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx768_bs1_latest_probe.jsonl", "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx768_bs1_latest_probe_trace_summary.json", "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx768_bs1_latest_probe_simulation_full.json"),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_share(summary: dict, region: str) -> float:
    for row in summary.get("regions", []):
        if row["region"] == region:
            return row["latency_share"]
    raise KeyError(region)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the full CPU/GPU suite with all current policies and build a merged dataset.")
    parser.add_argument("--env", default="jongyun_2026")
    parser.add_argument("--output-json", default="artifacts/reports/full_correlation_dataset.json")
    parser.add_argument("--output-csv", default="artifacts/reports/full_correlation_dataset.csv")
    args = parser.parse_args()

    rows = []
    for device_kind, model, context, batch_size, trace_path, summary_path, output_path in ENTRIES:
        subprocess.run(
            [
                "conda", "run", "-n", args.env, "python", "scripts/run_trace_simulation.py",
                "--trace", trace_path,
                "--output", output_path,
                "--min-offload-bytes", "32768",
            ],
            check=True,
            cwd=ROOT,
        )
        summary = load_json(ROOT / summary_path)
        sim = load_json(ROOT / output_path)
        results = {row["policy_name"]: row for row in sim["results"]}
        rows.append(
            {
                "device_kind": device_kind,
                "model": model,
                "context": context,
                "batch_size": batch_size,
                "attention_share": get_share(summary, "attention"),
                "mlp_share": get_share(summary, "mlp"),
                "online_speedup": results["online_predictor"]["speedup_vs_gpu_only"],
                "adaptive_feature_speedup": results["adaptive_feature"]["speedup_vs_gpu_only"],
                "adaptive_family_speedup": results["adaptive_family"]["speedup_vs_gpu_only"],
                "adaptive_score_speedup": results["adaptive_score"]["speedup_vs_gpu_only"],
                "oracle_speedup": results["oracle"]["speedup_vs_gpu_only"],
                "online_gap_closed": results["online_predictor"]["oracle_gap_closed"],
                "adaptive_feature_gap_closed": results["adaptive_feature"]["oracle_gap_closed"],
                "adaptive_family_gap_closed": results["adaptive_family"]["oracle_gap_closed"],
                "adaptive_score_gap_closed": results["adaptive_score"]["oracle_gap_closed"],
                "simulation_report": output_path,
            }
        )

    payload = {"num_points": len(rows), "rows": rows}
    out_json = ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_csv = ROOT / args.output_csv
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
