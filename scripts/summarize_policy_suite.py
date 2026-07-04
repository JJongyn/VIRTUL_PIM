#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize policy results across multiple simulation reports.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for input_path in args.inputs:
        payload = load_json(input_path)
        results = {row["policy_name"]: row for row in payload["results"]}
        row = {
            "file": input_path,
            "trace": payload.get("trace", ""),
            "model": payload.get("model", ""),
            "prompt_tokens_target": payload.get("prompt_tokens_target"),
            "batch_size": payload.get("batch_size"),
            "gpu_only_speedup": results["gpu_only"]["speedup_vs_gpu_only"],
            "online_predictor_speedup": results["online_predictor"]["speedup_vs_gpu_only"],
            "adaptive_score_speedup": results["adaptive_score"]["speedup_vs_gpu_only"],
            "oracle_speedup": results["oracle"]["speedup_vs_gpu_only"],
            "adaptive_gap_closed": results["adaptive_score"]["oracle_gap_closed"],
            "online_gap_closed": results["online_predictor"]["oracle_gap_closed"],
        }
        rows.append(row)

    summary = {
        "num_reports": len(rows),
        "rows": rows,
        "adaptive_matches_oracle": sum(
            1 for row in rows if abs(row["adaptive_score_speedup"] - row["oracle_speedup"]) < 1e-9
        ),
        "adaptive_beats_online": sum(
            1 for row in rows if row["adaptive_score_speedup"] > row["online_predictor_speedup"] + 1e-9
        ),
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
            f"{row['file']}: online={row['online_predictor_speedup']:.3f} "
            f"adaptive={row['adaptive_score_speedup']:.3f} "
            f"oracle={row['oracle_speedup']:.3f}"
        )


if __name__ == "__main__":
    main()
