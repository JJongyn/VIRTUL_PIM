#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


TRACES = [
    ("artifacts/traces/gpt2_none_ctx256_bs1_kvmech.jsonl", "artifacts/reports/gpt2_none_ctx256_bs1_kvmech_simulation.json"),
    ("artifacts/traces/gpt2_none_ctx512_bs1_kvmech.jsonl", "artifacts/reports/gpt2_none_ctx512_bs1_kvmech_simulation.json"),
    ("artifacts/traces/gpt2_none_ctx768_bs1_kvmech.jsonl", "artifacts/reports/gpt2_none_ctx768_bs1_kvmech_simulation.json"),
    ("artifacts/traces/qwen_qwen2_5_1_5b_instruct_4bit_ctx256_bs1_kvmech.jsonl", "artifacts/reports/qwen_qwen2_5_1_5b_instruct_4bit_ctx256_bs1_kvmech_simulation.json"),
    ("artifacts/traces/qwen_qwen2_5_1_5b_instruct_4bit_ctx512_bs1_kvmech.jsonl", "artifacts/reports/qwen_qwen2_5_1_5b_instruct_4bit_ctx512_bs1_kvmech_simulation.json"),
    ("artifacts/traces/qwen_qwen2_5_1_5b_instruct_4bit_ctx768_bs1_kvmech.jsonl", "artifacts/reports/qwen_qwen2_5_1_5b_instruct_4bit_ctx768_bs1_kvmech_simulation.json"),
    ("artifacts/traces/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx256_bs1_kvmech.jsonl", "artifacts/reports/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx256_bs1_kvmech_simulation.json"),
    ("artifacts/traces/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx512_bs1_kvmech.jsonl", "artifacts/reports/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx512_bs1_kvmech_simulation.json"),
    ("artifacts/traces/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx768_bs1_kvmech.jsonl", "artifacts/reports/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx768_bs1_kvmech_simulation.json"),
    ("artifacts/traces/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx256_bs1_kvmech.jsonl", "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx256_bs1_kvmech_simulation.json"),
    ("artifacts/traces/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx512_bs1_kvmech.jsonl", "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx512_bs1_kvmech_simulation.json"),
    ("artifacts/traces/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx768_bs1_kvmech.jsonl", "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx768_bs1_kvmech_simulation.json"),
    ("artifacts/traces/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx512_bs2_kvmech.jsonl", "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx512_bs2_kvmech_simulation.json"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the focused KV-mechanism suite with the latest policies.")
    parser.add_argument("--env", default="jongyun_2026")
    parser.add_argument("--min-offload-bytes", default="32768")
    args = parser.parse_args()

    for trace_path, output_path in TRACES:
        subprocess.run(
            [
                "conda", "run", "-n", args.env, "python", "scripts/run_trace_simulation.py",
                "--trace", trace_path,
                "--output", output_path,
                "--min-offload-bytes", args.min_offload_bytes,
            ],
            check=True,
            cwd=ROOT,
        )
        print(f"recomputed {output_path}")


if __name__ == "__main__":
    main()
