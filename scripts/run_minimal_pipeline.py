#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from papi_virtual.adapters.transformers_decode import TraceConfig, trace_transformers_decode
from papi_virtual.analysis import summarize_for_markdown, summarize_trace, write_region_csv, write_trace_summary
from papi_virtual.cost_model import VirtualPIMCostModel
from papi_virtual.profiler import load_trace
from papi_virtual.simulator import default_policies, simulate_policy
from papi_virtual.profiler import write_simulation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the smallest end-to-end PAPI virtual pipeline.")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--prompt", default="Explain why LLM decoding can become memory-bound.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trace-output", default="artifacts/traces/minimal_demo.jsonl")
    parser.add_argument("--summary-json", default="artifacts/reports/minimal_trace_summary.json")
    parser.add_argument("--summary-csv", default="artifacts/reports/minimal_trace_regions.csv")
    parser.add_argument("--summary-md", default="artifacts/reports/minimal_trace_summary.md")
    parser.add_argument("--simulation-output", default="artifacts/reports/minimal_simulation.json")
    args = parser.parse_args()

    trace_path = trace_transformers_decode(
        TraceConfig(
            model_name=args.model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            output_path=args.trace_output,
        )
    )

    records = load_trace(trace_path)
    payload = summarize_trace(records)
    write_trace_summary(args.summary_json, payload)
    write_region_csv(args.summary_csv, payload)
    Path(args.summary_md).write_text(summarize_for_markdown(payload), encoding="utf-8")

    cost_model = VirtualPIMCostModel()
    sim_rows = []
    for policy in default_policies():
        _, summary = simulate_policy(records, policy, cost_model)
        sim_rows.append(asdict(summary))

    write_simulation_report(
        args.simulation_output,
        {
            "trace": trace_path,
            "model": args.model,
            "device": args.device,
            "results": sim_rows,
        },
    )

    print(f"trace={trace_path}")
    print(f"trace_summary={args.summary_json}")
    print(f"simulation={args.simulation_output}")


if __name__ == "__main__":
    main()
