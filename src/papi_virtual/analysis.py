from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import csv
import json

from .schema import KernelRecord


def summarize_trace(records: list[KernelRecord]) -> dict:
    by_region: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "latency_ms": 0.0, "bytes_moved": 0.0, "flops": 0.0})
    contexts: list[int] = []
    batches: list[int] = []
    kv_bytes_values: list[float] = []
    kv_bytes_per_token_values: list[float] = []
    memory_pressure_values: list[float] = []

    for record in records:
        bucket = by_region[record.region]
        bucket["count"] += 1.0
        bucket["latency_ms"] += record.latency_ms
        bucket["bytes_moved"] += record.bytes_moved
        bucket["flops"] += record.flops
        contexts.append(record.context_len)
        batches.append(record.batch_size)
        if record.region == "attention":
            metadata = record.metadata or {}
            kv_bytes = float(metadata.get("kv_bytes_est", 0.0) or 0.0)
            kv_bytes_per_token = float(metadata.get("kv_bytes_per_token_est", 0.0) or 0.0)
            memory_pressure = float(metadata.get("memory_pressure_proxy", 0.0) or 0.0)
            if kv_bytes > 0.0:
                kv_bytes_values.append(kv_bytes)
            if kv_bytes_per_token > 0.0:
                kv_bytes_per_token_values.append(kv_bytes_per_token)
            if memory_pressure > 0.0:
                memory_pressure_values.append(memory_pressure)

    total_latency = sum(record.latency_ms for record in records)
    region_rows = []
    for region, bucket in sorted(by_region.items()):
        intensity = 0.0 if bucket["bytes_moved"] == 0 else bucket["flops"] / bucket["bytes_moved"]
        share = 0.0 if total_latency == 0 else bucket["latency_ms"] / total_latency
        region_rows.append(
            {
                "region": region,
                "count": int(bucket["count"]),
                "latency_ms": bucket["latency_ms"],
                "latency_share": share,
                "bytes_moved": bucket["bytes_moved"],
                "flops": bucket["flops"],
                "arithmetic_intensity": intensity,
            }
        )

    return {
        "num_records": len(records),
        "num_steps": len({record.step_idx for record in records}),
        "context_min": min(contexts) if contexts else 0,
        "context_max": max(contexts) if contexts else 0,
        "batch_min": min(batches) if batches else 0,
        "batch_max": max(batches) if batches else 0,
        "total_latency_ms": total_latency,
        "regions": region_rows,
        "mean_kv_bytes_est": sum(kv_bytes_values) / len(kv_bytes_values) if kv_bytes_values else 0.0,
        "mean_kv_bytes_per_token_est": sum(kv_bytes_per_token_values) / len(kv_bytes_per_token_values) if kv_bytes_per_token_values else 0.0,
        "mean_memory_pressure_proxy": sum(memory_pressure_values) / len(memory_pressure_values) if memory_pressure_values else 0.0,
    }


def write_trace_summary(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_region_csv(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "region",
                "count",
                "latency_ms",
                "latency_share",
                "bytes_moved",
                "flops",
                "arithmetic_intensity",
            ],
        )
        writer.writeheader()
        for row in payload.get("regions", []):
            writer.writerow(row)


def summarize_for_markdown(payload: dict) -> str:
    lines = [
        "# Trace Summary",
        "",
        f"- Records: {payload['num_records']}",
        f"- Steps: {payload['num_steps']}",
        f"- Context range: {payload['context_min']} .. {payload['context_max']}",
        f"- Batch range: {payload['batch_min']} .. {payload['batch_max']}",
        f"- Total latency (observed): {payload['total_latency_ms']:.3f} ms",
        f"- Mean KV bytes (attention records): {payload.get('mean_kv_bytes_est', 0.0):.1f}",
        f"- Mean KV bytes/token (attention records): {payload.get('mean_kv_bytes_per_token_est', 0.0):.3f}",
        f"- Mean memory-pressure proxy (attention records): {payload.get('mean_memory_pressure_proxy', 0.0):.3f}",
        "",
        "## Regions",
        "",
        "| Region | Count | Latency (ms) | Share | Intensity |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload.get("regions", []):
        lines.append(
            f"| {row['region']} | {row['count']} | {row['latency_ms']:.3f} | "
            f"{row['latency_share']:.3f} | {row['arithmetic_intensity']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)
