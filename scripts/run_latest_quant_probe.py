#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import re
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
    "Long-context LLM decoding repeatedly revisits attention states and KV cache data, "
    "so runtime pressure can shift from compute-dominated execution to memory-sensitive execution. "
)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


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
    parser = argparse.ArgumentParser(description="Run a latest-model quantized trace/analyze/simulate probe.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="4bit")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--min-offload-bytes", type=float, default=256 * 1024)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    model_slug = slugify(args.model)
    tag = f"_{slugify(args.tag)}" if args.tag else ""
    base = f"{model_slug}_{args.quantization}_ctx{args.prompt_tokens}_bs{args.batch_size}{tag}"

    trace_output = ROOT / "artifacts" / "traces" / f"{base}.jsonl"
    summary_json = ROOT / "artifacts" / "reports" / f"{base}_trace_summary.json"
    summary_csv = ROOT / "artifacts" / "reports" / f"{base}_trace_regions.csv"
    summary_md = ROOT / "artifacts" / "reports" / f"{base}_trace_summary.md"
    simulation_output = ROOT / "artifacts" / "reports" / f"{base}_simulation.json"

    prompt = build_long_prompt(args.model, args.prompt_tokens)
    trace_path = trace_transformers_decode(
        TraceConfig(
            model_name=args.model,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            device=args.device,
            torch_dtype=args.torch_dtype,
            quantization=args.quantization,
            output_path=str(trace_output),
        )
    )

    records = load_trace(trace_path)
    trace_summary = summarize_trace(records)
    write_trace_summary(str(summary_json), trace_summary)
    write_region_csv(str(summary_csv), trace_summary)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(summarize_for_markdown(trace_summary), encoding="utf-8")

    cost_model = VirtualPIMCostModel(VirtualPIMConfig(min_offload_bytes=args.min_offload_bytes))
    sim_rows = []
    for policy in default_policies():
        _, summary = simulate_policy(records, policy, cost_model)
        sim_rows.append(asdict(summary))

    write_simulation_report(
        str(simulation_output),
        {
            "trace": trace_path,
            "model": args.model,
            "device": args.device,
            "quantization": args.quantization,
            "prompt_tokens_target": args.prompt_tokens,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "min_offload_bytes": args.min_offload_bytes,
            "results": sim_rows,
        },
    )

    print(f"trace={trace_path}")
    print(f"trace_summary={summary_json}")
    print(f"simulation={simulation_output}")


if __name__ == "__main__":
    main()
