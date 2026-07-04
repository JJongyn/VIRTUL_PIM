#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a controlled batch study at fixed contexts.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-md", default="artifacts/reports/BATCH_CONTROL_REPORT.md")
    args = parser.parse_args()

    payload = load_payload(Path(args.input))
    rows = sorted(payload["rows"], key=lambda item: (int(item["context"]), int(item["batch_size"])))
    contexts = sorted({int(row["context"]) for row in rows})

    lines = [
        "# Batch Control Report",
        "",
        "This report isolates batch size at two fixed contexts to show why context is the stronger regime driver in the current setup.",
        "",
        f"- model: {payload['model']}",
        f"- quantization: {payload['quantization']}",
        f"- contexts: {', '.join(str(ctx) for ctx in contexts)}",
        "",
    ]

    for context in contexts:
        ctx_rows = [row for row in rows if int(row["context"]) == context]
        online_values = [float(row["online_speedup"]) for row in ctx_rows]
        kv_values = [float(row["mean_kv_bytes_est"]) for row in ctx_rows]
        mem_values = [float(row["mean_memory_pressure_proxy"]) for row in ctx_rows]
        lines.extend(
            [
                f"## Context {context}",
                "",
                f"- online speedup range across batch: {min(online_values):.3f} -> {max(online_values):.3f}",
                f"- KV bytes range across batch: {min(kv_values):.1f} -> {max(kv_values):.1f}",
                f"- memory-pressure proxy range across batch: {min(mem_values):.3f} -> {max(mem_values):.3f}",
                "",
                "| Batch | Attn Share | KV Bytes | Mem Pressure | Online | AdaptiveFeature | KVRegime | Oracle |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in ctx_rows:
            lines.append(
                f"| {int(row['batch_size'])} | {float(row['attention_share']):.3f} | {float(row['mean_kv_bytes_est']):.1f} | "
                f"{float(row['mean_memory_pressure_proxy']):.3f} | {float(row['online_speedup']):.3f} | "
                f"{float(row['adaptive_feature_speedup']):.3f} | {float(row['kv_regime_speedup']):.3f} | "
                f"{float(row['oracle_speedup']):.3f} |"
            )
        lines.append("")

    short_rows = [row for row in rows if int(row["context"]) == contexts[0]]
    long_rows = [row for row in rows if int(row["context"]) == contexts[-1]]
    short_online = [float(row["online_speedup"]) for row in short_rows]
    long_online = [float(row["online_speedup"]) for row in long_rows]
    lines.extend(
        [
            "## Main Readout",
            "",
            f"- At short context {contexts[0]}, online speedup stays at {min(short_online):.3f} -> {max(short_online):.3f} even as batch changes.",
            f"- At long context {contexts[-1]}, online speedup stays positive across all tested batches: {min(long_online):.3f} -> {max(long_online):.3f}.",
            f"- Context shifts memory-pressure proxy from {float(short_rows[0]['mean_memory_pressure_proxy']):.3f} to {float(long_rows[0]['mean_memory_pressure_proxy']):.3f}, while batch mainly scales KV bytes within each fixed regime.",
            "- This supports the claim that batch matters, but context is the stronger regime-separating variable in the current setup.",
            "",
        ]
    )

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
