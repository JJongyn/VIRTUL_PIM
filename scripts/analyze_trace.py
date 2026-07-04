#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from papi_virtual.analysis import summarize_for_markdown, summarize_trace, write_region_csv, write_trace_summary
from papi_virtual.profiler import load_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a decode trace before policy simulation.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--json-output", default="artifacts/reports/trace_summary.json")
    parser.add_argument("--csv-output", default="artifacts/reports/trace_regions.csv")
    parser.add_argument("--md-output", default="artifacts/reports/trace_summary.md")
    args = parser.parse_args()

    records = load_trace(args.trace)
    payload = summarize_trace(records)
    write_trace_summary(args.json_output, payload)
    write_region_csv(args.csv_output, payload)

    md = summarize_for_markdown(payload)
    md_path = Path(args.md_output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")

    print(f"wrote {args.json_output}")
    print(f"wrote {args.csv_output}")
    print(f"wrote {args.md_output}")


if __name__ == "__main__":
    main()

