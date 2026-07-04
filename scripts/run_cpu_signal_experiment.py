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
from papi_virtual.cost_model import VirtualPIMConfig, VirtualPIMCostModel
from papi_virtual.profiler import load_trace, write_simulation_report
from papi_virtual.simulator import default_policies, simulate_policy


BASE_SENTENCE = (
    "Large language model decoding repeatedly reads model weights and KV cache states, "
    "so performance can shift from compute bound to memory bound as sequence length grows. "
)


def build_long_prompt(model_name: str, target_tokens: int) -> str:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("This script requires transformers.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = BASE_SENTENCE
    while len(tokenizer(prompt, return_tensors="pt")["input_ids"][0]) < target_tokens:
        prompt += BASE_SENTENCE
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a longer-context CPU experiment to surface virtual PIM signal.")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-offload-bytes", type=float, default=32768.0)
    parser.add_argument("--trace-output", default="artifacts/traces/cpu_signal_probe.jsonl")
    parser.add_argument("--summary-json", default="artifacts/reports/cpu_signal_probe_trace_summary.json")
    parser.add_argument("--summary-csv", default="artifacts/reports/cpu_signal_probe_trace_regions.csv")
    parser.add_argument("--summary-md", default="artifacts/reports/cpu_signal_probe_trace_summary.md")
    parser.add_argument("--simulation-output", default="artifacts/reports/cpu_signal_probe_simulation.json")
    args = parser.parse_args()

    prompt = build_long_prompt(args.model, args.prompt_tokens)
    trace_path = trace_transformers_decode(
        TraceConfig(
            model_name=args.model,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            device=args.device,
            output_path=args.trace_output,
        )
    )

    records = load_trace(trace_path)
    trace_summary = summarize_trace(records)
    write_trace_summary(args.summary_json, trace_summary)
    write_region_csv(args.summary_csv, trace_summary)
    Path(args.summary_md).write_text(summarize_for_markdown(trace_summary), encoding="utf-8")

    cost_model = VirtualPIMCostModel(VirtualPIMConfig(min_offload_bytes=args.min_offload_bytes))
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
            "prompt_tokens_target": args.prompt_tokens,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "min_offload_bytes": args.min_offload_bytes,
            "results": sim_rows,
        },
    )

    print(f"trace={trace_path}")
    print(f"trace_summary={args.summary_json}")
    print(f"simulation={args.simulation_output}")


if __name__ == "__main__":
    main()
