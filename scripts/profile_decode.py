#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from papi_virtual.profiler import DecodeProfiler, synthetic_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a starter decode trace for virtual PIM experiments.")
    parser.add_argument("--output", default="artifacts/traces/synthetic_decode.jsonl")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=1024)
    args = parser.parse_args()

    profiler = DecodeProfiler(args.output)
    for record in synthetic_trace(num_steps=args.steps, base_batch=args.batch, base_context=args.context):
        profiler.add_record(record)
    profiler.dump()

    summary = profiler.summary()
    print(f"wrote {int(summary['records'])} records to {args.output}")
    print(f"total_latency_ms={summary['total_latency_ms']:.3f}")


if __name__ == "__main__":
    main()

