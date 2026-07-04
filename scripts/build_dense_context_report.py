#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def first_positive(rows: list[dict], key: str, threshold: float = 1.05) -> int | None:
    positives = [int(row["context"]) for row in rows if float(row[key]) >= threshold]
    return min(positives) if positives else None


def last_weak(rows: list[dict], key: str, threshold: float = 1.05) -> int | None:
    weak = [int(row["context"]) for row in rows if float(row[key]) < threshold]
    return max(weak) if weak else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize dense context sweeps for regime-transition analysis.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-md", default="artifacts/reports/DENSE_CONTEXT_REPORT.md")
    args = parser.parse_args()

    payloads = [load_payload(Path(path)) for path in args.inputs]
    lines = [
        "# Dense Context Report",
        "",
        "This report sharpens regime-transition boundaries with denser context sweeps.",
        "",
    ]

    for payload in payloads:
        rows = sorted(payload["rows"], key=lambda item: int(item["context"]))
        online_first = first_positive(rows, "online_speedup")
        online_last_weak = last_weak(rows, "online_speedup")
        kv_first = first_positive(rows, "kv_regime_speedup")
        kv_last_weak = last_weak(rows, "kv_regime_speedup")
        lines.extend(
            [
                f"## {payload['model']}",
                "",
                f"- first online-positive context: {online_first}",
                f"- last online-weak context: {online_last_weak}",
                f"- online transition window: {online_last_weak} -> {online_first}" if online_first is not None and online_last_weak is not None else "- online transition window: n/a",
                f"- first adaptive-feature-positive context: {first_positive(rows, 'adaptive_feature_speedup')}",
                f"- first kv-regime-positive context: {kv_first}",
                f"- last kv-regime-weak context: {kv_last_weak}",
                f"- kv-regime transition window: {kv_last_weak} -> {kv_first}" if kv_first is not None and kv_last_weak is not None else "- kv-regime transition window: n/a",
                "",
                "| Context | Attn Share | KV Bytes | Mem Pressure | Online | AdaptiveFeature | KVRegime | Oracle |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {int(row['context'])} | {float(row['attention_share']):.3f} | {float(row['mean_kv_bytes_est']):.1f} | "
                f"{float(row['mean_memory_pressure_proxy']):.3f} | {float(row['online_speedup']):.3f} | "
                f"{float(row['adaptive_feature_speedup']):.3f} | {float(row['kv_regime_speedup']):.3f} | {float(row['oracle_speedup']):.3f} |"
            )
        lines.append("")

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
