#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from papi_virtual.cost_model import VirtualPIMConfig, VirtualPIMCostModel
from papi_virtual.profiler import load_trace
from papi_virtual.simulator import default_policies, simulate_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep virtual PIM parameters on a fixed trace.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", default="artifacts/reports/pim_parameter_sweep.json")
    parser.add_argument("--pim-bws", default="2400,3200,4800,6400")
    parser.add_argument("--pim-tflops", default="18,36,72")
    parser.add_argument("--transfer-overheads-ms", default="0.02,0.01,0.005,0.001")
    parser.add_argument("--sync-overheads-ms", default="0.01,0.005,0.001,0.0")
    parser.add_argument("--min-offload-bytes", type=float, default=32768.0)
    args = parser.parse_args()

    records = load_trace(args.trace)
    pim_bws = [float(x) for x in args.pim_bws.split(",") if x]
    pim_tflops = [float(x) for x in args.pim_tflops.split(",") if x]
    transfer_overheads = [float(x) for x in args.transfer_overheads_ms.split(",") if x]
    sync_overheads = [float(x) for x in args.sync_overheads_ms.split(",") if x]

    sweep_rows = []
    for bw in pim_bws:
        for tflops in pim_tflops:
            for transfer_ms in transfer_overheads:
                for sync_ms in sync_overheads:
                    cost_model = VirtualPIMCostModel(
                        VirtualPIMConfig(
                            pim_peak_bw_gbps=bw,
                            pim_peak_tflops=tflops,
                            transfer_overhead_ms=transfer_ms,
                            sync_overhead_ms=sync_ms,
                            min_offload_bytes=args.min_offload_bytes,
                        )
                    )
                    row = {
                        "config": {
                            "pim_bw_gbps": bw,
                            "pim_tflops": tflops,
                            "transfer_overhead_ms": transfer_ms,
                            "sync_overhead_ms": sync_ms,
                            "min_offload_bytes": args.min_offload_bytes,
                        },
                        "results": [],
                    }
                    for policy in default_policies():
                        _, summary = simulate_policy(records, policy, cost_model)
                        row["results"].append(asdict(summary))
                    sweep_rows.append(row)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"trace": args.trace, "rows": sweep_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    best_rows = []
    for row in sweep_rows:
        best_policy = max(row["results"], key=lambda x: x["speedup_vs_gpu_only"])
        if best_policy["speedup_vs_gpu_only"] > 1.0:
            best_rows.append({"config": row["config"], "best_policy": best_policy})

    print(f"wrote {args.output}")
    print(f"num_sweep_rows={len(sweep_rows)}")
    print(f"num_positive_rows={len(best_rows)}")
    for item in best_rows[:10]:
        cfg = item["config"]
        pol = item["best_policy"]
        print(
            f"positive: policy={pol['policy_name']} speedup={pol['speedup_vs_gpu_only']:.3f} "
            f"bw={cfg['pim_bw_gbps']} tflops={cfg['pim_tflops']} "
            f"transfer={cfg['transfer_overhead_ms']} sync={cfg['sync_overhead_ms']}"
        )


if __name__ == "__main__":
    main()
