#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


def fmean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PIM-favorable regimes from the merged dataset.")
    parser.add_argument("--input-csv", default="artifacts/reports/full_correlation_dataset.csv")
    parser.add_argument("--output-md", default="artifacts/reports/REGIME_INSIGHT_REPORT.md")
    parser.add_argument("--online-threshold", type=float, default=1.05)
    args = parser.parse_args()

    rows = load_rows(Path(args.input_csv))
    for row in rows:
        for key in (
            "context",
            "batch_size",
            "attention_share",
            "online_speedup",
            "adaptive_feature_speedup",
            "adaptive_family_speedup",
            "oracle_speedup",
        ):
            row[key] = float(row[key])

    online_positive = [r for r in rows if r["online_speedup"] >= args.online_threshold]
    online_weak = [r for r in rows if r["online_speedup"] < args.online_threshold]
    adaptive_positive = [r for r in rows if r["adaptive_feature_speedup"] >= args.online_threshold]

    by_model: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    lines = [
        "# Regime Insight Report",
        "",
        f"- Total points: {len(rows)}",
        f"- Online-positive points (`online_speedup >= {args.online_threshold:.2f}`): {len(online_positive)}",
        f"- Online-weak points (`online_speedup < {args.online_threshold:.2f}`): {len(online_weak)}",
        f"- Adaptive-feature-positive points: {len(adaptive_positive)}",
        "",
        "## Positive vs Weak Regimes",
        "",
        f"- Mean context in online-positive regime: {fmean([r['context'] for r in online_positive]):.1f}",
        f"- Mean context in online-weak regime: {fmean([r['context'] for r in online_weak]):.1f}",
        f"- Mean attention share in online-positive regime: {fmean([r['attention_share'] for r in online_positive]):.3f}",
        f"- Mean attention share in online-weak regime: {fmean([r['attention_share'] for r in online_weak]):.3f}",
        f"- Mean oracle speedup in online-positive regime: {fmean([r['oracle_speedup'] for r in online_positive]):.3f}",
        f"- Mean oracle speedup in online-weak regime: {fmean([r['oracle_speedup'] for r in online_weak]):.3f}",
        "",
        "## Model-Family Readout",
        "",
    ]

    for model, model_rows in sorted(by_model.items()):
        positives = [r for r in model_rows if r["online_speedup"] >= args.online_threshold]
        first_positive_context = min((r["context"] for r in positives), default=None)
        lines.extend(
            [
                f"### {model}",
                "",
                f"- points: {len(model_rows)}",
                f"- mean online speedup: {fmean([r['online_speedup'] for r in model_rows]):.3f}",
                f"- mean attention share: {fmean([r['attention_share'] for r in model_rows]):.3f}",
                f"- first positive context: {first_positive_context if first_positive_context is not None else 'none'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Main Takeaways",
            "",
            "- Virtual PIM value is not universal; it appears in a subset of decode regimes.",
            "- Context remains the clearest empirical separator between weak and positive regimes.",
            "- Attention share is directionally useful but does not fully explain the regime boundary on its own.",
            "- Model family changes the onset of positive regimes, especially on newer quantized models.",
            "",
            "## Positive-Regime Points",
            "",
            "| Device | Model | Context | Batch | Attention Share | Online | AdaptiveFeature | Oracle |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in sorted(online_positive, key=lambda r: (str(r["model"]), r["context"], r["batch_size"])):
        lines.append(
            f"| {row['device_kind']} | {row['model']} | {int(row['context'])} | {int(row['batch_size'])} | "
            f"{row['attention_share']:.3f} | {row['online_speedup']:.3f} | {row['adaptive_feature_speedup']:.3f} | {row['oracle_speedup']:.3f} |"
        )
    lines.append("")

    output = Path(args.output_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
