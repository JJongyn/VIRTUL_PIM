#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from math import sqrt
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


FOCUSED_ENTRIES = [
    ("gpt2", 256, 1, "artifacts/reports/gpt2_none_ctx256_bs1_kvmech_trace_summary.json", "artifacts/reports/gpt2_none_ctx256_bs1_kvmech_simulation.json"),
    ("gpt2", 512, 1, "artifacts/reports/gpt2_none_ctx512_bs1_kvmech_trace_summary.json", "artifacts/reports/gpt2_none_ctx512_bs1_kvmech_simulation.json"),
    ("gpt2", 768, 1, "artifacts/reports/gpt2_none_ctx768_bs1_kvmech_trace_summary.json", "artifacts/reports/gpt2_none_ctx768_bs1_kvmech_simulation.json"),
    ("qwen2.5-1.5b-instruct-4bit", 256, 1, "artifacts/reports/qwen_qwen2_5_1_5b_instruct_4bit_ctx256_bs1_kvmech_trace_summary.json", "artifacts/reports/qwen_qwen2_5_1_5b_instruct_4bit_ctx256_bs1_kvmech_simulation.json"),
    ("qwen2.5-1.5b-instruct-4bit", 512, 1, "artifacts/reports/qwen_qwen2_5_1_5b_instruct_4bit_ctx512_bs1_kvmech_trace_summary.json", "artifacts/reports/qwen_qwen2_5_1_5b_instruct_4bit_ctx512_bs1_kvmech_simulation.json"),
    ("qwen2.5-1.5b-instruct-4bit", 768, 1, "artifacts/reports/qwen_qwen2_5_1_5b_instruct_4bit_ctx768_bs1_kvmech_trace_summary.json", "artifacts/reports/qwen_qwen2_5_1_5b_instruct_4bit_ctx768_bs1_kvmech_simulation.json"),
    ("qwen3-8b-4bit", 256, 1, "artifacts/reports/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx256_bs1_kvmech_trace_summary.json", "artifacts/reports/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx256_bs1_kvmech_simulation.json"),
    ("qwen3-8b-4bit", 512, 1, "artifacts/reports/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx512_bs1_kvmech_trace_summary.json", "artifacts/reports/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx512_bs1_kvmech_simulation.json"),
    ("qwen3-8b-4bit", 768, 1, "artifacts/reports/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx768_bs1_kvmech_trace_summary.json", "artifacts/reports/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx768_bs1_kvmech_simulation.json"),
    ("smollm2-1.7b-instruct-4bit", 256, 1, "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx256_bs1_kvmech_trace_summary.json", "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx256_bs1_kvmech_simulation.json"),
    ("smollm2-1.7b-instruct-4bit", 512, 1, "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx512_bs1_kvmech_trace_summary.json", "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx512_bs1_kvmech_simulation.json"),
    ("smollm2-1.7b-instruct-4bit", 768, 1, "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx768_bs1_kvmech_trace_summary.json", "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx768_bs1_kvmech_simulation.json"),
    ("smollm2-1.7b-instruct-4bit", 512, 2, "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx512_bs2_kvmech_trace_summary.json", "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx512_bs2_kvmech_simulation.json"),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return 0.0
    return cov / sqrt(var_x * var_y)


def sim_result_map(sim: dict) -> dict[str, dict]:
    return {row["policy_name"]: row for row in sim["results"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a KV-cache / memory-pressure mechanism report from focused GPU traces.")
    parser.add_argument("--output-json", default="artifacts/reports/kv_mechanism_dataset.json")
    parser.add_argument("--output-csv", default="artifacts/reports/kv_mechanism_dataset.csv")
    parser.add_argument("--output-md", default="artifacts/reports/KV_MECHANISM_REPORT.md")
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    for model, context, batch_size, summary_path, sim_path in FOCUSED_ENTRIES:
        summary = load_json(ROOT / summary_path)
        sim = load_json(ROOT / sim_path)
        sim_rows = sim_result_map(sim)
        rows.append(
            {
                "model": model,
                "context": context,
                "batch_size": batch_size,
                "attention_share": float(next(r["latency_share"] for r in summary["regions"] if r["region"] == "attention")),
                "mean_kv_bytes_est": float(summary.get("mean_kv_bytes_est", 0.0)),
                "mean_kv_bytes_per_token_est": float(summary.get("mean_kv_bytes_per_token_est", 0.0)),
                "mean_memory_pressure_proxy": float(summary.get("mean_memory_pressure_proxy", 0.0)),
                "online_speedup": float(sim_rows["online_predictor"]["speedup_vs_gpu_only"]),
                "adaptive_feature_speedup": float(sim_rows["adaptive_feature"]["speedup_vs_gpu_only"]),
                "kv_regime_speedup": float(sim_rows["kv_regime"]["speedup_vs_gpu_only"]),
                "oracle_speedup": float(sim_rows["oracle"]["speedup_vs_gpu_only"]),
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

    xs_context = [float(row["context"]) for row in rows]
    xs_kv = [float(row["mean_kv_bytes_est"]) for row in rows]
    xs_kv_per_token = [float(row["mean_kv_bytes_per_token_est"]) for row in rows]
    xs_mem = [float(row["mean_memory_pressure_proxy"]) for row in rows]
    ys_online = [float(row["online_speedup"]) for row in rows]
    ys_kv_regime = [float(row["kv_regime_speedup"]) for row in rows]
    ys_oracle = [float(row["oracle_speedup"]) for row in rows]
    batch1_ctx512 = [row for row in rows if int(row["context"]) == 512 and int(row["batch_size"]) == 1]
    batch2_ctx512 = [row for row in rows if int(row["context"]) == 512 and int(row["batch_size"]) == 2]
    by_model: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    kv_better_than_adaptive = sum(
        1 for row in rows if float(row["kv_regime_speedup"]) > float(row["adaptive_feature_speedup"]) + 1e-9
    )
    kv_equal_adaptive = sum(
        1 for row in rows if abs(float(row["kv_regime_speedup"]) - float(row["adaptive_feature_speedup"])) <= 1e-9
    )

    lines = [
        "# KV Mechanism Report",
        "",
        f"- Focused GPU points: {len(rows)}",
        "",
        "## Correlations",
        "",
        f"- context vs mean_kv_bytes_est: {corr(xs_context, xs_kv):.3f}",
        f"- context vs online_speedup: {corr(xs_context, ys_online):.3f}",
        f"- mean_kv_bytes_est vs online_speedup: {corr(xs_kv, ys_online):.3f}",
        f"- mean_kv_bytes_est vs kv_regime_speedup: {corr(xs_kv, ys_kv_regime):.3f}",
        f"- mean_kv_bytes_per_token_est vs online_speedup: {corr(xs_kv_per_token, ys_online):.3f}",
        f"- memory_pressure_proxy vs online_speedup: {corr(xs_mem, ys_online):.3f}",
        f"- memory_pressure_proxy vs kv_regime_speedup: {corr(xs_mem, ys_kv_regime):.3f}",
        f"- mean_kv_bytes_est vs oracle_speedup: {corr(xs_kv, ys_oracle):.3f}",
        f"- memory_pressure_proxy vs oracle_speedup: {corr(xs_mem, ys_oracle):.3f}",
        "",
        "## Main Readout",
        "",
        "- This report tests the mechanism story directly on refreshed GPU traces with explicit KV proxies.",
        "- If context tracks KV bytes strongly and KV bytes track speedup positively, the long-context story is better grounded than a context-only correlation.",
        "- Memory-pressure proxy is expected to be model-family dependent; the same context can yield different positive-regime onsets.",
        "- The `kv_regime` policy uses only KV-related runtime features and avoids direct cost-model internals.",
        f"- `kv_regime` beats `adaptive_feature` on {kv_better_than_adaptive}/{len(rows)} focused points and ties on {kv_equal_adaptive}/{len(rows)}.",
        f"- At matched context 512, mean batch-1 online speedup is {sum(float(r['online_speedup']) for r in batch1_ctx512) / len(batch1_ctx512):.3f} "
        f"vs {sum(float(r['online_speedup']) for r in batch2_ctx512) / len(batch2_ctx512):.3f} for batch-2, which supports the earlier observation that batch is a weaker driver than context in the current setup.",
        "",
        "## Family-Matched Onset",
        "",
    ]
    for model, model_rows in sorted(by_model.items()):
        ordered = sorted(model_rows, key=lambda item: (int(item["context"]), int(item["batch_size"])))
        online_positive = [r for r in ordered if float(r["online_speedup"]) > 1.05]
        adaptive_positive = [r for r in ordered if float(r["adaptive_feature_speedup"]) > 1.05]
        kv_positive = [r for r in ordered if float(r["kv_regime_speedup"]) > 1.05]
        lines.append(
            f"- `{model}`: first online-positive context = {min((int(r['context']) for r in online_positive), default=-1)}, "
            f"first adaptive-feature-positive context = {min((int(r['context']) for r in adaptive_positive), default=-1)}, "
            f"first kv-regime-positive context = {min((int(r['context']) for r in kv_positive), default=-1)}"
        )
    lines.extend(
        [
            "",
            "## Points",
            "",
            "| Model | Context | Batch | Attn Share | KV Bytes | KV Bytes/Token | Mem Pressure | Online | AdaptiveFeature | KVRegime | Oracle |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["model"]), int(item["context"]), int(item["batch_size"]))):
        lines.append(
            f"| {row['model']} | {int(row['context'])} | {int(row['batch_size'])} | "
            f"{float(row['attention_share']):.3f} | {float(row['mean_kv_bytes_est']):.1f} | "
            f"{float(row['mean_kv_bytes_per_token_est']):.3f} | {float(row['mean_memory_pressure_proxy']):.3f} | "
            f"{float(row['online_speedup']):.3f} | {float(row['adaptive_feature_speedup']):.3f} | "
            f"{float(row['kv_regime_speedup']):.3f} | {float(row['oracle_speedup']):.3f} |"
        )
    lines.append("")

    out_md = ROOT / args.output_md
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
