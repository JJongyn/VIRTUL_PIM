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

from papi_virtual.cost_model import VirtualPIMConfig, VirtualPIMCostModel
from papi_virtual.profiler import load_trace, write_simulation_report
from papi_virtual.simulator import default_policies, simulate_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate virtual PIM dispatch policies on a trace.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", default="artifacts/reports/simulation_report.json")
    parser.add_argument("--pim-bw-gbps", type=float, default=2400.0)
    parser.add_argument("--pim-tflops", type=float, default=18.0)
    parser.add_argument("--transfer-overhead-ms", type=float, default=0.020)
    parser.add_argument("--sync-overhead-ms", type=float, default=0.010)
    parser.add_argument("--min-offload-bytes", type=float, default=256 * 1024)
    args = parser.parse_args()

    records = load_trace(args.trace)
    cost_model = VirtualPIMCostModel(
        VirtualPIMConfig(
            pim_peak_bw_gbps=args.pim_bw_gbps,
            pim_peak_tflops=args.pim_tflops,
            transfer_overhead_ms=args.transfer_overhead_ms,
            sync_overhead_ms=args.sync_overhead_ms,
            min_offload_bytes=args.min_offload_bytes,
        )
    )

    results = []
    for policy in default_policies():
        _, summary = simulate_policy(records, policy, cost_model)
        results.append(asdict(summary))

    payload = {
        "trace": args.trace,
        "num_records": len(records),
        "config": {
            "pim_bw_gbps": args.pim_bw_gbps,
            "pim_tflops": args.pim_tflops,
            "transfer_overhead_ms": args.transfer_overhead_ms,
            "sync_overhead_ms": args.sync_overhead_ms,
            "min_offload_bytes": args.min_offload_bytes,
        },
        "results": results,
    }
    write_simulation_report(args.output, payload)

    print(f"wrote report to {args.output}")
    for row in results:
        print(
            f"{row['policy_name']}: speedup={row['speedup_vs_gpu_only']:.3f} "
            f"gap_closed={row['oracle_gap_closed']:.3f} pim_dispatches={row['dispatches_to_pim']}"
        )


if __name__ == "__main__":
    main()
