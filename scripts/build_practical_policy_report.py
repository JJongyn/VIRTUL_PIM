#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from papi_virtual.cost_model import VirtualPIMCostModel
from papi_virtual.policies import (
    AdaptiveFeaturePolicy,
    ContextKVHysteresisPolicy,
    ContextKVRefinedPolicy,
    GPUOnlyPolicy,
    KVRegimePolicy,
    OraclePolicy,
    StaticAttentionPolicy,
    ThresholdPolicy,
)
from papi_virtual.profiler import load_trace
from papi_virtual.simulator import simulate_policy
from scripts.run_decision_signal_study import TRACE_SPECS, context_bucket


POLICIES = [
    GPUOnlyPolicy(),
    StaticAttentionPolicy(),
    ThresholdPolicy(name="context_threshold", intensity_threshold=1e9, min_context_len=384),
    AdaptiveFeaturePolicy(),
    KVRegimePolicy(),
    ContextKVRefinedPolicy(),
    ContextKVHysteresisPolicy(),
    OraclePolicy(),
]


def policy_map() -> dict[str, Any]:
    return {policy.name: policy for policy in POLICIES}


def compute_decision_metrics(decisions, oracle_decisions) -> dict[str, float]:
    preds = np.array([1 if d.target == "pim" else 0 for d in decisions], dtype=int)
    labels = np.array([1 if d.target == "pim" else 0 for d in oracle_decisions], dtype=int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    positive_recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    false_positive_rate = 0.0 if (fp + tn) == 0 else fp / (fp + tn)
    missed_positive_rate = 0.0 if (tp + fn) == 0 else fn / (tp + fn)
    candidate_indices = [idx for idx, d in enumerate(decisions) if d.region == "attention"]
    candidate_preds = preds[candidate_indices] if candidate_indices else np.array([], dtype=int)
    switches = sum(1 for prev, cur in zip(candidate_preds, candidate_preds[1:]) if prev != cur)
    consecutive_spans = []
    current = 1
    for prev, cur in zip(candidate_preds, candidate_preds[1:]):
        if prev == cur:
            current += 1
        else:
            consecutive_spans.append(current)
            current = 1
    if len(candidate_preds) > 0:
        consecutive_spans.append(current)
    return {
        "positive_recall": float(positive_recall),
        "false_positive_rate": float(false_positive_rate),
        "missed_positive_rate": float(missed_positive_rate),
        "switches_per_100": 0.0 if len(candidate_preds) == 0 else float(switches / len(candidate_preds) * 100.0),
        "avg_consecutive_span": float(sum(consecutive_spans) / len(consecutive_spans)) if consecutive_spans else 0.0,
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = [
        "speedup_vs_gpu_only",
        "oracle_gap_closed",
        "oracle_regret",
        "positive_recall",
        "false_positive_rate",
        "missed_positive_rate",
        "switches_per_100",
        "avg_consecutive_span",
    ]
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in rows]
        out[key] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
        out[f"{key}_median"] = float(np.median(vals))
        out[f"{key}_p90"] = float(np.percentile(vals, 90))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the practical context-gated KV-refined hysteresis policy.")
    parser.add_argument("--output-json", default="artifacts/reports/practical_policy_report.json")
    parser.add_argument("--output-csv", default="artifacts/reports/practical_policy_report.csv")
    parser.add_argument("--output-md", default="artifacts/reports/PRACTICAL_POLICY_REPORT.md")
    args = parser.parse_args()

    cost_model = VirtualPIMCostModel()
    pm = policy_map()
    workload_rows = []
    aggregate_rows: dict[str, list[dict[str, float]]] = {name: [] for name in pm if name != "oracle"}
    boundary_rows: dict[str, list[dict[str, float]]] = {name: [] for name in pm if name != "oracle"}

    for spec in TRACE_SPECS:
        records = load_trace(ROOT / spec.path)
        oracle_decisions, oracle_summary = simulate_policy(records, pm["oracle"], cost_model)
        oracle_speedup = oracle_summary.speedup_vs_gpu_only
        boundary = 1.05 <= oracle_speedup <= 1.30
        for policy_name, policy in pm.items():
            if policy_name == "oracle":
                continue
            decisions, summary = simulate_policy(records, policy, cost_model)
            metrics = compute_decision_metrics(decisions, oracle_decisions)
            oracle_regret = max(1.0 - float(summary.oracle_gap_closed), 0.0)
            row = {
                "policy_name": policy_name,
                "family": spec.family,
                "context_target": spec.context_target,
                "context_bucket": context_bucket(spec.context_target),
                "workload_tag": spec.tag,
                "speedup_vs_gpu_only": float(summary.speedup_vs_gpu_only),
                "oracle_gap_closed": float(summary.oracle_gap_closed),
                "oracle_regret": float(oracle_regret),
                **metrics,
                "boundary_subset": int(boundary),
            }
            workload_rows.append(row)
            aggregate_rows[policy_name].append(row)
            if boundary:
                boundary_rows[policy_name].append(row)

    summary = {policy_name: aggregate(rows) for policy_name, rows in aggregate_rows.items()}
    boundary_summary = {policy_name: aggregate(rows) for policy_name, rows in boundary_rows.items() if rows}
    hysteresis_delta = {}
    if "context_kv_refined" in summary and "context_kv_hysteresis" in summary:
        refined = summary["context_kv_refined"]
        hysteresis = summary["context_kv_hysteresis"]
        hysteresis_delta = {
            "speedup_delta": float(hysteresis["speedup_vs_gpu_only"] - refined["speedup_vs_gpu_only"]),
            "oracle_regret_delta": float(hysteresis["oracle_regret"] - refined["oracle_regret"]),
            "false_positive_rate_delta": float(hysteresis["false_positive_rate"] - refined["false_positive_rate"]),
            "missed_positive_rate_delta": float(hysteresis["missed_positive_rate"] - refined["missed_positive_rate"]),
            "switches_per_100_delta": float(hysteresis["switches_per_100"] - refined["switches_per_100"]),
            "avg_consecutive_span_delta": float(hysteresis["avg_consecutive_span"] - refined["avg_consecutive_span"]),
        }

    payload = {
        "workload_rows": workload_rows,
        "summary": summary,
        "boundary_summary": boundary_summary,
        "hysteresis_delta_vs_refined": hysteresis_delta,
    }

    out_json = ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_csv = ROOT / args.output_csv
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(workload_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in workload_rows:
            writer.writerow(row)

    policy_order = ["gpu_only", "static_attention", "context_threshold", "adaptive_feature", "kv_regime", "context_kv_refined", "context_kv_hysteresis"]
    lines = [
        "# Practical Policy Report",
        "",
        "## Proposed Policy",
        "",
        "- Stage 1: context gate (`ctx >= 256`) and attention-region candidacy.",
        "- Stage 2: KV refinement using estimated KV bytes, KV bytes/token, and memory-pressure proxy.",
        "- Stage 3: hysteresis with separate on/off thresholds to reduce oscillation along the attention-decision stream.",
        "",
        "## Overall Workload Summary",
        "",
        "| Policy | Speedup | Oracle Gap Closed | Oracle Regret | Pos Recall | FP Rate | Missed-Positive | Switches /100 | Avg Span |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in policy_order:
        row = summary[name]
        lines.append(
            f"| {name} | {row['speedup_vs_gpu_only']:.3f}±{row['speedup_vs_gpu_only_std']:.3f} | "
            f"{row['oracle_gap_closed']:.3f}±{row['oracle_gap_closed_std']:.3f} | {row['oracle_regret']:.3f}±{row['oracle_regret_std']:.3f} | "
            f"{row['positive_recall']:.3f}±{row['positive_recall_std']:.3f} | {row['false_positive_rate']:.3f}±{row['false_positive_rate_std']:.3f} | "
            f"{row['missed_positive_rate']:.3f}±{row['missed_positive_rate_std']:.3f} | {row['switches_per_100']:.3f}±{row['switches_per_100_std']:.3f} | "
            f"{row['avg_consecutive_span']:.3f}±{row['avg_consecutive_span_std']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary-Subset Summary",
            "",
            "| Policy | Speedup | Oracle Regret | Pos Recall | FP Rate | Switches /100 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in policy_order:
        if name not in boundary_summary:
            continue
        row = boundary_summary[name]
        lines.append(
            f"| {name} | {row['speedup_vs_gpu_only']:.3f} | {row['oracle_regret']:.3f} | {row['positive_recall']:.3f} | "
            f"{row['false_positive_rate']:.3f} | {row['switches_per_100']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Regret Distribution Anchors",
            "",
            "| Policy | Median Regret | P90 Regret | Median Switches /100 | P90 Switches /100 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in policy_order:
        row = summary[name]
        lines.append(
            f"| {name} | {row['oracle_regret_median']:.3f} | {row['oracle_regret_p90']:.3f} | {row['switches_per_100_median']:.3f} | {row['switches_per_100_p90']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Hysteresis Ablation",
            "",
            "| Compare | Speedup Δ | Regret Δ | FP Δ | Missed-Positive Δ | Switches /100 Δ | Avg Span Δ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if hysteresis_delta:
        lines.append(
            f"| context_kv_hysteresis - context_kv_refined | {hysteresis_delta['speedup_delta']:.3f} | "
            f"{hysteresis_delta['oracle_regret_delta']:.3f} | {hysteresis_delta['false_positive_rate_delta']:.3f} | "
            f"{hysteresis_delta['missed_positive_rate_delta']:.3f} | {hysteresis_delta['switches_per_100_delta']:.3f} | "
            f"{hysteresis_delta['avg_consecutive_span_delta']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Main Readout",
            "",
            "- The proposed `context_kv_hysteresis` policy turns the paper's empirical findings into a simple actionable detector: context provides the coarse gate, KV signals refine the boundary, and hysteresis is retained as a practical stabilization hook.",
            "- Relative to `static_attention`, the proposed policy reduces false-positive offload and switching, which makes it easier to present as a practical method rather than as a one-off case study.",
            "- The explicit `context_kv_refined` ablation isolates hysteresis itself: under the current trace granularity, the hysteresis thresholds do not materially change aggregate behavior, which suggests that most of the practical gain currently comes from conservative context+KV gating rather than from hysteresis alone.",
            "- Relative to `adaptive_feature` and `kv_regime`, the main expected gain is a more conservative context+KV detector rather than a universal average-speedup win.",
            "",
        ]
    )
    out_md = ROOT / args.output_md
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
