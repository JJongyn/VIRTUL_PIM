#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


PRESETS = {
    "conservative": {
        "pim_bw_gbps": 2000.0,
        "pim_tflops": 14.0,
        "transfer_overhead_ms": 0.030,
        "sync_overhead_ms": 0.015,
    },
    "default": {
        "pim_bw_gbps": 2400.0,
        "pim_tflops": 18.0,
        "transfer_overhead_ms": 0.020,
        "sync_overhead_ms": 0.010,
    },
    "optimistic": {
        "pim_bw_gbps": 2800.0,
        "pim_tflops": 24.0,
        "transfer_overhead_ms": 0.012,
        "sync_overhead_ms": 0.006,
    },
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small sensitivity study over the virtual PIM cost model.")
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--env", default="jongyun_2026")
    parser.add_argument("--min-offload-bytes", default="32768")
    parser.add_argument("--output-json", default="artifacts/reports/cost_sensitivity.json")
    parser.add_argument("--output-md", default="artifacts/reports/COST_SENSITIVITY_REPORT.md")
    args = parser.parse_args()

    rows = []
    for trace in args.traces:
        trace_name = Path(trace).name
        for preset_name, preset in PRESETS.items():
            output = ROOT / "artifacts" / "reports" / f"{Path(trace).stem}_{preset_name}_sensitivity_simulation.json"
            cmd = [
                "conda", "run", "-n", args.env, "python", "scripts/run_trace_simulation.py",
                "--trace", trace,
                "--output", str(output),
                "--min-offload-bytes", str(args.min_offload_bytes),
                "--pim-bw-gbps", str(preset["pim_bw_gbps"]),
                "--pim-tflops", str(preset["pim_tflops"]),
                "--transfer-overhead-ms", str(preset["transfer_overhead_ms"]),
                "--sync-overhead-ms", str(preset["sync_overhead_ms"]),
            ]
            subprocess.run(cmd, check=True, cwd=ROOT)
            sim = load_json(output)
            sim_rows = {row["policy_name"]: row for row in sim["results"]}
            rows.append(
                {
                    "trace": trace_name,
                    "preset": preset_name,
                    "online_speedup": sim_rows["online_predictor"]["speedup_vs_gpu_only"],
                    "kv_regime_speedup": sim_rows["kv_regime"]["speedup_vs_gpu_only"],
                    "adaptive_feature_speedup": sim_rows["adaptive_feature"]["speedup_vs_gpu_only"],
                    "oracle_speedup": sim_rows["oracle"]["speedup_vs_gpu_only"],
                }
            )

    payload = {"rows": rows}
    out_json = ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Cost Sensitivity Report",
        "",
        "This report checks whether the qualitative weak-regime / positive-regime split survives conservative, default, and optimistic virtual-PIM settings.",
        "",
        "| Trace | Preset | Online | AdaptiveFeature | KVRegime | Oracle |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['trace']} | {row['preset']} | {row['online_speedup']:.3f} | "
            f"{row['adaptive_feature_speedup']:.3f} | {row['kv_regime_speedup']:.3f} | {row['oracle_speedup']:.3f} |"
        )
    lines.append("")
    out_md = ROOT / args.output_md
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
