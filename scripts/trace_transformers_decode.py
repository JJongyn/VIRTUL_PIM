#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from papi_virtual.adapters.transformers_decode import TraceConfig, trace_transformers_decode


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a coarse decode trace from a Transformers model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="none")
    parser.add_argument("--output", default="artifacts/traces/transformers_decode.jsonl")
    args = parser.parse_args()

    path = trace_transformers_decode(
        TraceConfig(
            model_name=args.model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            torch_dtype=args.torch_dtype,
            quantization=args.quantization,
            output_path=args.output,
        )
    )
    print(f"wrote trace to {path}")


if __name__ == "__main__":
    main()
