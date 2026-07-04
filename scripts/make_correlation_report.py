#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs) ** 0.5
    deny = sum((y - my) ** 2 for y in ys) ** 0.5
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a compact markdown report from the full correlation dataset.")
    parser.add_argument("--input-csv", default="artifacts/reports/full_correlation_dataset.csv")
    parser.add_argument("--output-md", default="artifacts/reports/CORRELATION_REPORT.md")
    args = parser.parse_args()

    rows = []
    with Path(args.input_csv).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    attention = [float(r["attention_share"]) for r in rows]
    online = [float(r["online_speedup"]) for r in rows]
    adaptive_feature = [float(r["adaptive_feature_speedup"]) for r in rows]
    adaptive_family = [float(r["adaptive_family_speedup"]) for r in rows]
    oracle = [float(r["oracle_speedup"]) for r in rows]

    cpu_rows = [r for r in rows if r["device_kind"] == "cpu"]
    gpu_rows = [r for r in rows if r["device_kind"] == "gpu"]

    lines = [
        "# Correlation Report",
        "",
        f"- Points: {len(rows)}",
        f"- CPU points: {len(cpu_rows)}",
        f"- GPU points: {len(gpu_rows)}",
        "",
        "## Correlations",
        "",
        f"- attention_share vs online_speedup: {corr(attention, online):.3f}",
        f"- attention_share vs adaptive_feature_speedup: {corr(attention, adaptive_feature):.3f}",
        f"- attention_share vs adaptive_family_speedup: {corr(attention, adaptive_family):.3f}",
        f"- attention_share vs oracle_speedup: {corr(attention, oracle):.3f}",
        "",
        "## Notes",
        "",
        "- Higher attention share is associated with higher dynamic-offload gain in the current dataset.",
        "- `adaptive_feature` is the conservative realistic feature-only policy candidate.",
        "- `adaptive_family` adds model-family-aware priors while staying feature-driven.",
        "- `adaptive_score` remains useful as an oracle-friendly prototype but should not be treated as a deployment policy.",
        "",
        "## Rows",
        "",
        "| Device | Model | Context | Batch | Attn Share | Online | AdaptiveFeature | AdaptiveFamily | Oracle |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['device_kind']} | {r['model']} | {r['context']} | {r['batch_size']} | "
            f"{float(r['attention_share']):.3f} | {float(r['online_speedup']):.3f} | "
            f"{float(r['adaptive_feature_speedup']):.3f} | {float(r['adaptive_family_speedup']):.3f} | {float(r['oracle_speedup']):.3f} |"
        )
    lines.append("")

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
