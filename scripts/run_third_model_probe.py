#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a compact third-model probe on CPU and GPU.")
    parser.add_argument("--model", default="EleutherAI/pythia-160m")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--env", default="jongyun_2026")
    args = parser.parse_args()

    tag = args.model.replace("/", "_").replace("-", "_")
    cmds = [
        [
            "conda", "run", "-n", args.env, "python", "scripts/run_cpu_signal_experiment.py",
            "--model", args.model,
            "--device", "cpu",
            "--prompt-tokens", str(args.context),
            "--batch-size", "1",
            "--max-new-tokens", str(args.max_new_tokens),
            "--min-offload-bytes", "32768",
            "--trace-output", f"artifacts/traces/{tag}_cpu_ctx{args.context}.jsonl",
            "--summary-json", f"artifacts/reports/{tag}_cpu_ctx{args.context}_trace_summary.json",
            "--summary-csv", f"artifacts/reports/{tag}_cpu_ctx{args.context}_trace_regions.csv",
            "--summary-md", f"artifacts/reports/{tag}_cpu_ctx{args.context}_trace_summary.md",
            "--simulation-output", f"artifacts/reports/{tag}_cpu_ctx{args.context}_simulation.json",
        ],
        [
            "conda", "run", "-n", args.env, "python", "scripts/run_cpu_signal_experiment.py",
            "--model", args.model,
            "--device", "cuda:0",
            "--prompt-tokens", str(args.context),
            "--batch-size", "1",
            "--max-new-tokens", str(args.max_new_tokens),
            "--min-offload-bytes", "32768",
            "--trace-output", f"artifacts/traces/{tag}_gpu_ctx{args.context}.jsonl",
            "--summary-json", f"artifacts/reports/{tag}_gpu_ctx{args.context}_trace_summary.json",
            "--summary-csv", f"artifacts/reports/{tag}_gpu_ctx{args.context}_trace_regions.csv",
            "--summary-md", f"artifacts/reports/{tag}_gpu_ctx{args.context}_trace_summary.md",
            "--simulation-output", f"artifacts/reports/{tag}_gpu_ctx{args.context}_simulation.json",
        ],
    ]

    for cmd in cmds:
        subprocess.run(cmd, check=True, cwd=ROOT)

    print(f"completed third-model probe for {args.model}")


if __name__ == "__main__":
    main()
