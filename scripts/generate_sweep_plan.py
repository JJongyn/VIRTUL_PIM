#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import json


DEFAULT_BATCHES = [1, 2, 4, 8]
DEFAULT_CONTEXTS = [512, 2048, 8192]
DEFAULT_GEN_LENGTHS = [128, 512]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a pre-run sweep plan for decode profiling.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="artifacts/plans/sweep_plan.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    runs = []
    run_id = 0
    for batch, context, gen_len in product(DEFAULT_BATCHES, DEFAULT_CONTEXTS, DEFAULT_GEN_LENGTHS):
        runs.append(
            {
                "run_id": run_id,
                "model": args.model,
                "device": args.device,
                "batch_size": batch,
                "prompt_tokens_target": context,
                "max_new_tokens": gen_len,
                "trace_output": f"artifacts/traces/run_{run_id:03d}.jsonl",
                "summary_output": f"artifacts/reports/run_{run_id:03d}_trace_summary.json",
                "simulation_output": f"artifacts/reports/run_{run_id:03d}_simulation.json",
                "status": "planned",
            }
        )
        run_id += 1

    payload = {
        "model": args.model,
        "device": args.device,
        "num_runs": len(runs),
        "runs": runs,
    }

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} with {len(runs)} planned runs")


if __name__ == "__main__":
    main()

