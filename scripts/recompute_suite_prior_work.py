#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def run_sim(trace_path: str, output_path: str, env_name: str) -> dict:
    cmd = [
        "conda", "run", "-n", env_name, "python", "scripts/run_trace_simulation.py",
        "--trace", trace_path,
        "--output", output_path,
        "--min-offload-bytes", "32768",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    return json.loads(Path(ROOT / output_path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute key simulation suites under the prior-work-grounded cost model.")
    parser.add_argument("--env", default="jongyun_2026")
    parser.add_argument("--output", default="artifacts/reports/prior_work_suite_summary.json")
    args = parser.parse_args()

    trace_groups = [
        ("gpt2_ctx256", "artifacts/traces/gpt2_ctx256.jsonl", "artifacts/reports/gpt2_ctx256_simulation_prior_work.json"),
        ("gpt2_ctx512", "artifacts/traces/gpt2_ctx512.jsonl", "artifacts/reports/gpt2_ctx512_simulation_prior_work.json"),
        ("gpt2_ctx768", "artifacts/traces/gpt2_ctx768.jsonl", "artifacts/reports/gpt2_ctx768_simulation_prior_work.json"),
        ("gpt2_ctx960", "artifacts/traces/gpt2_ctx960.jsonl", "artifacts/reports/gpt2_ctx960_simulation_prior_work.json"),
        ("gpt2_ctx512_bs4", "artifacts/traces/gpt2_ctx512_bs4.jsonl", "artifacts/reports/gpt2_ctx512_bs4_simulation_prior_work.json"),
        ("opt125m_ctx256", "artifacts/traces/facebook_opt_125m_ctx256.jsonl", "artifacts/reports/opt125m_ctx256_simulation_prior_work.json"),
        ("opt125m_ctx512", "artifacts/traces/facebook_opt_125m_ctx512.jsonl", "artifacts/reports/opt125m_ctx512_simulation_prior_work.json"),
        ("opt125m_ctx768", "artifacts/traces/facebook_opt_125m_ctx768.jsonl", "artifacts/reports/opt125m_ctx768_simulation_prior_work.json"),
        ("opt125m_ctx512_bs4", "artifacts/traces/facebook_opt_125m_ctx512_bs4.jsonl", "artifacts/reports/opt125m_ctx512_bs4_simulation_prior_work.json"),
    ]

    rows = []
    for tag, trace, output in trace_groups:
        payload = run_sim(trace, output, args.env)
        results = {row["policy_name"]: row for row in payload["results"]}
        rows.append(
            {
                "tag": tag,
                "trace": trace,
                "report": output,
                "gpu_only": results["gpu_only"]["speedup_vs_gpu_only"],
                "online_predictor": results["online_predictor"]["speedup_vs_gpu_only"],
                "adaptive_score": results["adaptive_score"]["speedup_vs_gpu_only"],
                "oracle": results["oracle"]["speedup_vs_gpu_only"],
                "online_gap_closed": results["online_predictor"]["oracle_gap_closed"],
                "adaptive_gap_closed": results["adaptive_score"]["oracle_gap_closed"],
            }
        )

    summary = {
        "num_reports": len(rows),
        "rows": rows,
        "adaptive_matches_oracle": sum(1 for row in rows if abs(row["adaptive_score"] - row["oracle"]) < 1e-9),
        "adaptive_beats_online": sum(1 for row in rows if row["adaptive_score"] > row["online_predictor"] + 1e-9),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(
        f"num_reports={summary['num_reports']} "
        f"adaptive_matches_oracle={summary['adaptive_matches_oracle']} "
        f"adaptive_beats_online={summary['adaptive_beats_online']}"
    )
    for row in rows:
        print(
            f"{row['tag']}: online={row['online_predictor']:.3f} "
            f"adaptive={row['adaptive_score']:.3f} "
            f"oracle={row['oracle']:.3f}"
        )


if __name__ == "__main__":
    main()
