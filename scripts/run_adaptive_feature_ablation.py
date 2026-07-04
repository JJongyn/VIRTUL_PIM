#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from papi_virtual.cost_model import VirtualPIMCostModel
from papi_virtual.policies import AdaptiveFeaturePolicy, OraclePolicy
from papi_virtual.profiler import load_trace
from papi_virtual.simulator import simulate_policy


ABLATIONS = [
    ("full", {}),
    ("no_memory_pressure", {"use_memory_pressure": False}),
    ("no_context", {"use_context": False}),
    ("no_batch", {"use_batch": False}),
    ("no_bytes", {"use_bytes": False}),
    ("no_intensity", {"use_intensity": False}),
    ("no_latency", {"use_latency": False}),
    ("no_region_bias", {"use_region_bias": False}),
]


TRACES = [
    ("cpu_gpt2_512", "artifacts/traces/gpt2_ctx512.jsonl"),
    ("cpu_gpt2_960", "artifacts/traces/gpt2_ctx960.jsonl"),
    ("cpu_opt125m_512", "artifacts/traces/facebook_opt_125m_ctx512.jsonl"),
    ("gpu_gpt2_512", "artifacts/traces/gpt2_gpu_ctx512_bs1.jsonl"),
    ("gpu_gpt2_960", "artifacts/traces/gpt2_gpu_ctx960_bs1.jsonl"),
    ("gpu_opt125m_512", "artifacts/traces/opt125m_gpu_ctx512_bs1.jsonl"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate the realistic adaptive feature policy.")
    parser.add_argument("--output-json", default="artifacts/reports/adaptive_feature_ablation.json")
    parser.add_argument("--output-csv", default="artifacts/reports/adaptive_feature_ablation.csv")
    args = parser.parse_args()

    cost_model = VirtualPIMCostModel()
    rows = []
    for trace_tag, trace_path in TRACES:
        records = load_trace(trace_path)
        _, oracle_summary = simulate_policy(records, OraclePolicy(), cost_model)
        for ablation_name, overrides in ABLATIONS:
            policy = AdaptiveFeaturePolicy(**overrides)
            _, summary = simulate_policy(records, policy, cost_model)
            rows.append(
                {
                    "trace_tag": trace_tag,
                    "ablation": ablation_name,
                    "speedup": summary.speedup_vs_gpu_only,
                    "gap_closed": summary.oracle_gap_closed,
                    "oracle_speedup": oracle_summary.speedup_vs_gpu_only,
                }
            )

    payload = {"rows": rows}
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
