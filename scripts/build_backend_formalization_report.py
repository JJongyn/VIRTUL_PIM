#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from papi_virtual.cost_model import VirtualPIMConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a backend formalization report and LaTeX tables for the virtual PIM model.")
    parser.add_argument("--output-md", default="artifacts/reports/BACKEND_FORMALIZATION.md")
    parser.add_argument("--output-json", default="artifacts/reports/backend_formalization.json")
    parser.add_argument("--output-measured-table", default="final/tables/backend_measured_vs_modeled.tex")
    parser.add_argument("--output-params-table", default="final/tables/backend_parameters.tex")
    args = parser.parse_args()

    cfg = VirtualPIMConfig()
    payload = {
        "gpu_latency_formula": "L_gpu(r) = measured_trace_latency(r) if available, else max(FLOPs/TFLOPS_gpu, Bytes/BW_gpu).",
        "pim_latency_formula": (
            "L_pim(r) = max(L_gpu * compute_share * TFLOPS_gpu/TFLOPS_pim_eff, "
            "L_gpu * memory_share * BW_gpu/BW_pim_eff) + transfer_overhead + sync_overhead."
        ),
        "effective_modifiers": {
            "attention_long_context_bw_bonus": cfg.attention_long_context_bw_bonus,
            "fc_parallelism_penalty": cfg.fc_parallelism_penalty,
            "non_attention_penalty": cfg.non_attention_penalty,
        },
        "parameters": {
            "gpu_peak_bw_gbps": cfg.gpu_peak_bw_gbps,
            "gpu_peak_tflops": cfg.gpu_peak_tflops,
            "pim_peak_bw_gbps": cfg.pim_peak_bw_gbps,
            "pim_peak_tflops": cfg.pim_peak_tflops,
            "transfer_overhead_ms": cfg.transfer_overhead_ms,
            "sync_overhead_ms": cfg.sync_overhead_ms,
            "min_offload_bytes": cfg.min_offload_bytes,
        },
        "trace_measured": [
            "latency_ms",
            "context_len",
            "batch_size",
            "region",
        ],
        "trace_derived": [
            "bytes_moved",
            "flops",
            "arithmetic_intensity",
            "kv_bytes_est",
            "kv_bytes_per_token_est",
            "memory_pressure_proxy",
        ],
        "modeled_only": [
            "gpu_compute_ms",
            "gpu_memory_ms",
            "compute_share",
            "memory_share",
            "effective_pim_bw",
            "effective_pim_tflops",
            "estimated_pim_latency_ms",
        ],
        "calibration_anchor": (
            "The GPU side is anchored to trace-measured kernel latency whenever the trace provides latency_ms; "
            "the roofline terms are used only to decompose that measured latency into compute- and memory-dominant shares "
            "before applying the virtual PIM scaling."
        ),
        "parameter_sources": [
            "GPU roofline anchor (A100 312 TFLOPS / 1935 GB/s) follows the prior-work setup referenced in the code comments and paper discussion of PAPI-style grounding.",
            "PIM bandwidth is set above the GPU anchor (2400 vs 1935 GB/s) to model a memory-centric accelerator advantage without assuming GPU-class compute throughput.",
            "PIM throughput is intentionally low (18 TFLOP/s) to encode a conservative memory-centric backend and to avoid overstating gains on compute-dominant regions.",
            "Attention long-context bandwidth bonus and MLP/non-attention penalties are qualitative priors used to encode the intended memory-vs-compute asymmetry, not hardware-calibrated constants.",
            "The attention bonus rewards long-context memory sensitivity, while FC and non-attention penalties are deliberately chosen to avoid overestimating benefit outside attention-heavy regions.",
            "Transfer and sync overheads are explicit constant terms and are varied again in the sensitivity study rather than treated as fixed hardware facts.",
        ],
    }

    out_json = ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    measured_table = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Virtual backend inputs grouped by provenance.}",
            "\\label{tab:backend-measured-modeled}",
            "\\begin{tabular}{p{0.22\\linewidth}p{0.37\\linewidth}p{0.27\\linewidth}}",
            "\\toprule",
            "Category & Fields & Source / Interpretation \\\\",
            "\\midrule",
            "Trace-measured & region label, measured latency, context length, batch size & Directly observed in the collected decode traces \\\\",
            "Trace-derived & bytes moved, FLOPs, arithmetic intensity, estimated KV bytes, KV bytes/token, memory-pressure proxy & Derived from runtime metadata and trace-side estimators \\\\",
            "Modeled only & roofline compute/memory shares, effective PIM bandwidth/throughput modifiers, transfer/sync overheads, estimated PIM latency & Virtual replay assumptions used only for what-if comparison \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    out_measured = ROOT / args.output_measured_table
    out_measured.parent.mkdir(parents=True, exist_ok=True)
    out_measured.write_text(measured_table + "\n", encoding="utf-8")

    params_table = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Default virtual backend parameters, role, and provenance.}",
            "\\label{tab:backend-params}",
            "\\begin{tabular}{p{0.28\\linewidth}p{0.13\\linewidth}p{0.24\\linewidth}p{0.24\\linewidth}}",
            "\\toprule",
            "Parameter & Default & Role & Provenance \\\\",
            "\\midrule",
            f"GPU peak BW & {cfg.gpu_peak_bw_gbps:.0f} GB/s & Roofline anchor for memory time & Prior-work A100-style roofline anchor \\\\",
            f"GPU peak throughput & {cfg.gpu_peak_tflops:.0f} TFLOP/s & Roofline anchor for compute time & Prior-work A100-style roofline anchor \\\\",
            f"PIM peak BW & {cfg.pim_peak_bw_gbps:.0f} GB/s & Baseline virtual PIM memory rate & Memory-centric default: modestly above GPU BW to encode bandwidth advantage without assuming equal compute \\\\",
            f"PIM peak throughput & {cfg.pim_peak_tflops:.0f} TFLOP/s & Baseline virtual PIM compute rate & Conservative low-throughput default to avoid overstating compute-heavy gains \\\\",
            f"Transfer overhead & {cfg.transfer_overhead_ms:.3f} ms & Per-dispatch orchestration cost & Conservative midpoint varied again in preset sensitivity \\\\",
            f"Sync overhead & {cfg.sync_overhead_ms:.3f} ms & Per-dispatch synchronization cost & Conservative midpoint varied again in preset sensitivity \\\\",
            f"Attention BW bonus & {cfg.attention_long_context_bw_bonus:.2f} & Long-context attention memory modifier & Qualitative prior reflecting stronger KV pressure at long context \\\\",
            f"FC parallelism penalty & {cfg.fc_parallelism_penalty:.2f} & MLP parallelism penalty & Deliberate penalty to avoid overstating non-attention and MLP benefit \\\\",
            f"Non-attention penalty & {cfg.non_attention_penalty:.2f} & Throughput penalty outside attention & Deliberate penalty to keep non-attention gains conservative \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    out_params = ROOT / args.output_params_table
    out_params.parent.mkdir(parents=True, exist_ok=True)
    out_params.write_text(params_table + "\n", encoding="utf-8")

    lines = [
        "# Backend Formalization",
        "",
        "## Latency Model",
        "",
        "- GPU latency estimate:",
        "  - `L_gpu(r) = latency_ms(r)` when the trace already contains a measured region latency.",
        "  - Otherwise `L_gpu(r) = max(FLOPs / TFLOPS_gpu, Bytes / BW_gpu)`.",
        "- PIM latency estimate:",
        "  - `L_pim(r) = max(L_gpu * compute_share * TFLOPS_gpu / TFLOPS_pim_eff, L_gpu * memory_share * BW_gpu / BW_pim_eff) + transfer + sync`.",
        "- Effective modifiers:",
        f"  - Attention long-context bandwidth bonus: `{cfg.attention_long_context_bw_bonus:.2f}` scaled by `min(context / 1024, 1)`.",
        f"  - MLP parallelism penalty: `{cfg.fc_parallelism_penalty:.2f}` scaled by `min(decode_parallelism / 32, 1)`.",
        f"  - Non-attention throughput penalty: `{cfg.non_attention_penalty:.2f}`.",
        "",
        "## Provenance Split",
        "",
        "- Trace-measured: region labels, measured latency, context length, batch size.",
        "- Trace-derived: bytes moved, FLOPs, arithmetic intensity, estimated KV bytes, KV bytes per token, memory-pressure proxy.",
        "- Modeled only: roofline compute/memory shares, effective PIM bandwidth/throughput, transfer and sync overheads, estimated PIM latency.",
        "",
        "## Calibration Anchor",
        "",
        f"- {payload['calibration_anchor']}",
        "",
        "## Parameter Source Notes",
        "",
    ]
    for item in payload["parameter_sources"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- This backend is suitable for trace-grounded replay, signal comparison, and sensitivity analysis.",
            "- It is not a cycle-accurate simulator and should not be used to claim deployment-ready hardware speedups.",
            "- The paper should therefore treat backend-derived policy differences as evidence about predictive decision signals under the current model, not as universal scheduler truths.",
            "",
        ]
    )
    out_md = ROOT / args.output_md
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_measured_table}")
    print(f"wrote {args.output_params_table}")


if __name__ == "__main__":
    main()
