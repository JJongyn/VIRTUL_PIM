#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "final"
FIG_DIR = FINAL_DIR / "figures"
TABLE_DIR = FINAL_DIR / "tables"

matplotlib.rcParams.update(
    {
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.usetex": False,
        "mathtext.fontset": "stix",
    }
)

COLORS = {
    "Qwen2.5-1.5B": "#1f77b4",
    "Qwen3-8B": "#d62728",
    "SmolLM2-1.7B": "#2ca02c",
    "Phi-3.5-mini": "#ff7f0e",
}
MARKERS = {
    "Qwen2.5-1.5B": "o",
    "Qwen3-8B": "s",
    "SmolLM2-1.7B": "^",
    "Phi-3.5-mini": "D",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{suffix}")
    plt.close(fig)


def dense_payloads() -> list[tuple[str, list[dict]]]:
    payloads = [
        ("Qwen2.5-1.5B", load_json(ROOT / "artifacts/reports/qwen2_5_dense_context_sweep.json")["rows"]),
        ("Qwen3-8B", load_json(ROOT / "artifacts/reports/qwen3_dense_context_sweep.json")["rows"]),
        ("SmolLM2-1.7B", load_json(ROOT / "artifacts/reports/smollm2_dense_context_sweep.json")["rows"]),
    ]
    phi_rows = []
    for ctx in (256, 384, 768):
        summary = load_json(ROOT / f"artifacts/reports/microsoft_phi_3_5_mini_instruct_4bit_ctx{ctx}_bs1_phi35_probe_trace_summary.json")
        sim = load_json(ROOT / f"artifacts/reports/microsoft_phi_3_5_mini_instruct_4bit_ctx{ctx}_bs1_phi35_probe_simulation.json")
        sim_rows = {row["policy_name"]: row for row in sim["results"]}
        phi_rows.append(
            {
                "context": ctx,
                "attention_share": next(r["latency_share"] for r in summary["regions"] if r["region"] == "attention"),
                "mean_kv_bytes_est": summary.get("mean_kv_bytes_est", 0.0),
                "mean_memory_pressure_proxy": summary.get("mean_memory_pressure_proxy", 0.0),
                "online_speedup": sim_rows["online_predictor"]["speedup_vs_gpu_only"],
                "adaptive_feature_speedup": sim_rows["adaptive_feature"]["speedup_vs_gpu_only"],
                "kv_regime_speedup": sim_rows["kv_regime"]["speedup_vs_gpu_only"],
                "oracle_speedup": sim_rows["oracle"]["speedup_vs_gpu_only"],
            }
        )
    payloads.append(("Phi-3.5-mini", phi_rows))
    return payloads


def decision_signal_payload() -> dict:
    return load_json(ROOT / "artifacts/reports/decision_signal_study.json")


def backend_validation_payload() -> dict:
    return load_json(ROOT / "artifacts/reports/backend_validation.json")


def practical_policy_payload() -> dict:
    return load_json(ROOT / "artifacts/reports/practical_policy_report.json")


def generate_context_transition() -> None:
    fig, ax = plt.subplots(figsize=(6.1, 3.5))
    ax.axhspan(1.05, 1.40, color="#eef7ec", zorder=0)
    ax.axhspan(0.95, 1.05, color="#f8f1e8", zorder=0)
    ax.axhline(1.05, color="black", linewidth=1.0, linestyle="--")

    for label, rows in dense_payloads():
        rows = sorted(rows, key=lambda item: int(item["context"]))
        xs = [int(row["context"]) for row in rows]
        ys = [float(row["online_speedup"]) for row in rows]
        ax.plot(xs, ys, marker=MARKERS[label], markersize=5.5, linewidth=2.0, color=COLORS[label], label=label)

    ax.text(140, 1.217, "positive regime", fontsize=8, color="#3d6b35")
    ax.text(140, 1.014, "weak / borderline", fontsize=8, color="#8b5a2b")
    ax.set_xlim(110, 1045)
    ax.set_ylim(0.97, 1.24)
    ax.set_xticks([128, 256, 384, 512, 640, 768, 1024])
    ax.set_xlabel("Prompt Context Length")
    ax.set_ylabel("Online Speedup vs GPU-Only")
    ax.legend(loc="lower right", ncol=2, frameon=False, columnspacing=1.0, handletextpad=0.5)
    fig.tight_layout()
    save_fig(fig, "paper_backbone_context_transition")


def generate_kv_pressure_scatter() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.axhline(1.05, color="black", linewidth=1.0, linestyle="--")

    for label, rows in dense_payloads():
        xs = [float(row["mean_memory_pressure_proxy"]) for row in rows]
        ys = [float(row["online_speedup"]) for row in rows]
        sizes = [30 + int(row["context"]) / 12 for row in rows]
        ax.scatter(
            xs,
            ys,
            s=sizes,
            alpha=0.84,
            color=COLORS[label],
            marker=MARKERS[label],
            edgecolor="white",
            linewidth=0.7,
            label=label,
        )

    family_legend = ax.legend(loc="lower right", frameon=False, ncol=2, columnspacing=1.0, handletextpad=0.5)
    ax.add_artist(family_legend)
    size_handles = []
    for context, size in ((256, 52), (512, 72), (768, 94)):
        size_handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#9e9e9e", markeredgecolor="white", markersize=size ** 0.5, label=f"ctx={context}")
        )
    ax.legend(handles=size_handles, loc="upper left", frameon=False, title="Marker size")
    ax.set_xlim(0.992, 1.0002)
    ax.set_ylim(0.97, 1.24)
    ax.set_xlabel("Memory Pressure Proxy")
    ax.set_ylabel("Online Speedup vs GPU-Only")
    fig.tight_layout()
    save_fig(fig, "paper_backbone_kv_pressure_vs_speedup")


def generate_same_context_family() -> None:
    fig, ax1 = plt.subplots(figsize=(6.3, 3.5))
    labels = [
        "Qwen2.5\n4b",
        "Qwen2.5\n8b",
        "Qwen3\n4b",
        "SmolLM2\n4b",
        "SmolLM2\n8b",
        "Phi-3.5\n4b",
    ]
    kv_mib = [
        1655808 / (1024 * 1024),
        1655808 / (1024 * 1024),
        4415488 / (1024 * 1024),
        2232320 / (1024 * 1024),
        2232320 / (1024 * 1024),
        3643392 / (1024 * 1024),
    ]
    online = [1.000, 1.000, 1.156, 1.146, 1.151, 1.140]
    colors = [
        COLORS["Qwen2.5-1.5B"],
        COLORS["Qwen2.5-1.5B"],
        COLORS["Qwen3-8B"],
        COLORS["SmolLM2-1.7B"],
        COLORS["SmolLM2-1.7B"],
        COLORS["Phi-3.5-mini"],
    ]
    hatches = ["", "//", "", "", "//", ""]
    xs = np.arange(len(labels))
    bars = ax1.bar(xs, kv_mib, color=colors, edgecolor="white", linewidth=0.8)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
    ax1.set_ylabel("Estimated KV Bytes (MiB)")
    ax1.set_xlabel("Family at ctx=256")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    ax2.plot(xs, online, color="black", marker="o", linewidth=1.8)
    ax2.axhline(1.05, color="black", linewidth=1.0, linestyle="--")
    ax2.set_ylabel("Online Speedup vs GPU-Only")
    ax2.set_ylim(0.98, 1.19)

    legend_handles = [
        Line2D([0], [0], color=COLORS["Qwen2.5-1.5B"], lw=6, label="Qwen2.5"),
        Line2D([0], [0], color=COLORS["Qwen3-8B"], lw=6, label="Qwen3"),
        Line2D([0], [0], color=COLORS["SmolLM2-1.7B"], lw=6, label="SmolLM2"),
        Line2D([0], [0], color=COLORS["Phi-3.5-mini"], lw=6, label="Phi-3.5"),
        Line2D([0], [0], color="white", marker="s", markerfacecolor="white", markeredgecolor="gray", markersize=8, linestyle="None", label="hatched = 8-bit"),
        Line2D([0], [0], color="black", marker="o", linewidth=1.8, label="online speedup"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", frameon=False, ncol=2, columnspacing=0.8, handlelength=1.8)
    fig.tight_layout()
    save_fig(fig, "paper_same_context_family")


def generate_policy_comparison() -> None:
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    labels = ["Qwen2.5\nctx256", "Qwen2.5\nctx768", "Qwen3\nctx256", "SmolLM2\nctx256", "Phi-3.5\nctx256", "Phi-3.5\nctx384"]
    values = {
        "static_attention": [1.000, 1.194, 1.155, 1.150, 1.140, 1.181],
        "adaptive_feature": [1.000, 1.194, 1.000, 1.000, 1.000, 1.181],
        "kv_regime": [1.000, 1.194, 1.155, 1.150, 1.140, 1.181],
        "oracle": [1.274, 1.337, 1.277, 1.276, 1.276, 1.297],
    }
    policy_order = ["static_attention", "adaptive_feature", "kv_regime", "oracle"]
    policy_colors = ["#8c8c8c", "#4c78a8", "#f58518", "#54a24b"]
    width = 0.18
    xs = list(range(len(labels)))
    for idx, (policy, color) in enumerate(zip(policy_order, policy_colors)):
        offsets = [x + (idx - 1.5) * width for x in xs]
        ax.bar(offsets, values[policy], width=width, color=color, label=policy)
    ax.axhline(1.05, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylim(0.95, 1.36)
    ax.set_ylabel("Speedup vs GPU-Only")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left", ncol=2, frameon=False)
    fig.tight_layout()
    save_fig(fig, "paper_policy_comparison")


def generate_negative_cases() -> None:
    payload = decision_signal_payload()
    workload_rows = payload["workload_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    online_speedups = [float(row["online_speedup"]) for row in workload_rows]
    axes[0].hist(online_speedups, bins=[0.98, 1.01, 1.04, 1.07, 1.10, 1.14, 1.18, 1.22], color="#4c78a8", edgecolor="white")
    axes[0].axvline(1.05, color="black", linewidth=1.0, linestyle="--")
    axes[0].set_xlabel("Online Speedup vs GPU-Only")
    axes[0].set_ylabel("Workloads")
    axes[0].set_title("Gain Distribution")

    contexts = sorted({int(row["context_target"]) for row in workload_rows})
    online_fraction = []
    adaptive_fraction = []
    kv_fraction = []
    for context in contexts:
        ctx_rows = [row for row in workload_rows if int(row["context_target"]) == context]
        online_fraction.append(sum(int(row["online_positive"]) for row in ctx_rows) / len(ctx_rows))
        adaptive_fraction.append(sum(int(row["adaptive_positive"]) for row in ctx_rows) / len(ctx_rows))
        kv_fraction.append(sum(int(row["kv_regime_positive"]) for row in ctx_rows) / len(ctx_rows))
    axes[1].plot(contexts, online_fraction, marker="o", linewidth=1.8, color="#4c78a8", label="online")
    axes[1].plot(contexts, adaptive_fraction, marker="s", linewidth=1.8, color="#8c8c8c", label="adaptive_feature")
    axes[1].plot(contexts, kv_fraction, marker="^", linewidth=1.8, color="#f58518", label="kv_regime")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_xticks(contexts)
    axes[1].set_xlabel("Prompt Context Length")
    axes[1].set_ylabel("Positive Fraction")
    axes[1].set_title("Positive-Regime Fraction")
    axes[1].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save_fig(fig, "paper_negative_cases")


def generate_decision_signal_study() -> None:
    payload = decision_signal_payload()
    family_summary = payload["split_results"]["summary"]["family_held_out"]
    context_summary = payload["split_results"]["summary"]["context_bucket_held_out"]
    labels = ["attn", "ctx", "ctx+attn", "kv", "all"]
    keys = ["attention_only", "context_only", "context_plus_attention", "kv_only", "all_features"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    width = 0.36
    xs = list(range(len(labels)))
    family_f1 = [float(family_summary[key]["f1"]) for key in keys]
    family_f1_std = [float(family_summary[key]["f1_std"]) for key in keys]
    family_fp = [float(family_summary[key]["false_positive_rate"]) for key in keys]
    axes[0].bar([x - width / 2 for x in xs], family_f1, width=width, color="#4c78a8", label="F1", yerr=family_f1_std, capsize=3)
    axes[0].bar([x + width / 2 for x in xs], family_fp, width=width, color="#b3b3b3", label="FP rate")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Family-Held-Out")
    axes[0].legend(loc="lower left", frameon=False)

    context_f1 = [float(context_summary[key]["f1"]) for key in keys]
    context_f1_std = [float(context_summary[key]["f1_std"]) for key in keys]
    context_auc = [float(context_summary[key]["roc_auc"]) for key in keys]
    axes[1].bar([x - width / 2 for x in xs], context_f1, width=width, color="#f58518", label="F1", yerr=context_f1_std, capsize=3)
    axes[1].bar([x + width / 2 for x in xs], context_auc, width=width, color="#54a24b", label="ROC-AUC")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Context-Bucket-Held-Out")
    axes[1].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save_fig(fig, "paper_decision_signal_study")


def generate_backend_validation_figure() -> None:
    backend = backend_validation_payload()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    presets = PRESETS = ["conservative", "default", "optimistic"]
    xs = list(range(len(presets)))
    online = [float(backend["sensitivity"]["by_preset"][p]["online_positive_fraction"]) for p in presets]
    adaptive = [float(backend["sensitivity"]["by_preset"][p]["adaptive_positive_fraction"]) for p in presets]
    kv = [float(backend["sensitivity"]["by_preset"][p]["kv_positive_fraction"]) for p in presets]
    axes[0].plot(xs, online, marker="o", linewidth=1.8, color="#4c78a8", label="online")
    axes[0].plot(xs, adaptive, marker="s", linewidth=1.8, color="#8c8c8c", label="adaptive")
    axes[0].plot(xs, kv, marker="^", linewidth=1.8, color="#f58518", label="kv")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(["cons.", "default", "opt."])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Positive Fraction")
    axes[0].set_title("Backend Preset Stability")
    axes[0].legend(loc="upper left", frameon=False)

    onset = backend["sensitivity"]["onset_by_preset"]
    families = [("gpt2", "#9ecae1"), ("Qwen2.5", COLORS["Qwen2.5-1.5B"]), ("Qwen3", COLORS["Qwen3-8B"]), ("Phi-3.5", COLORS["Phi-3.5-mini"])]
    for family, color in families:
        ys = []
        for preset in presets:
            value = onset[preset].get(family)
            ys.append(np.nan if value is None else float(value))
        axes[1].plot(xs, ys, marker="o", linewidth=1.8, color=color, label=family)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(["cons.", "default", "opt."])
    axes[1].set_ylim(120, 820)
    axes[1].set_ylabel("First Online-Positive Context")
    axes[1].set_title("Onset Shift Across Presets")
    axes[1].legend(loc="upper left", frameon=False, ncol=2, fontsize=7)
    fig.tight_layout()
    save_fig(fig, "paper_backend_validation")


def generate_family_onset_summary() -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    families = ["gpt2", "opt-125m", "Qwen2.5", "Qwen3", "SmolLM2", "Phi-3.5"]
    onset = [512, 512, 384, 256, 256, 256]
    colors = ["#9ecae1", "#6baed6", COLORS["Qwen2.5-1.5B"], COLORS["Qwen3-8B"], COLORS["SmolLM2-1.7B"], COLORS["Phi-3.5-mini"]]
    ax.barh(families, onset, color=colors)
    ax.set_xlabel("First Online-Positive Context")
    ax.set_xlim(0, 560)
    fig.tight_layout()
    save_fig(fig, "paper_family_onset_summary")


def generate_sensitivity_summary() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    labels = ["gpt2\n256", "gpt2\n768", "Qwen2.5\n256", "Qwen2.5\n768", "Phi-3.5\n256", "Phi-3.5\n768"]
    conservative = [1.000, 1.080, 1.000, 1.096, 1.046, 1.087]
    default = [1.000, 1.160, 1.000, 1.194, 1.140, 1.175]
    optimistic = [1.000, 1.224, 1.000, 1.276, 1.219, 1.247]
    xs = list(range(len(labels)))
    ax.plot(xs, conservative, marker="o", linewidth=1.8, color="#8c8c8c", label="conservative")
    ax.plot(xs, default, marker="s", linewidth=1.8, color="#4c78a8", label="default")
    ax.plot(xs, optimistic, marker="^", linewidth=1.8, color="#54a24b", label="optimistic")
    ax.axhline(1.05, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylim(0.95, 1.30)
    ax.set_ylabel("Online Speedup vs GPU-Only")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left", ncol=3, frameon=False)
    fig.tight_layout()
    save_fig(fig, "paper_sensitivity_summary")


def build_framework_mermaid() -> None:
    mermaid = "\n".join(
        [
            "flowchart LR",
            "    A[Real LLM Decode Run] --> B[Trace Adapter]",
            "    B --> C[Region-Level Trace]",
            "    C --> D[Trace Analysis]",
            "    C --> E[Virtual PIM Cost Model]",
            "    D --> F[Runtime Features\\ncontext, bytes, intensity, KV proxy]",
            "    E --> G[Policy Simulator]",
            "    F --> G",
            "    G --> H[Policy Comparison\\nGPU-only, static, online, KV-aware, oracle]",
            "    H --> I[Regime Maps and Mechanism Figures]",
        ]
    )
    mmd = FINAL_DIR / "framework_overview.mmd"
    md = FINAL_DIR / "framework_overview.md"
    mmd.write_text(mermaid + "\n", encoding="utf-8")
    md.write_text("# Framework Overview\n\n```mermaid\n" + mermaid + "\n```\n", encoding="utf-8")

    png = FINAL_DIR / "framework_overview.png"
    if shutil.which("mmdc"):
        subprocess.run(["mmdc", "-i", str(mmd), "-o", str(png), "-b", "transparent"], check=False, cwd=ROOT)


def write_tables() -> None:
    main_table = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Representative regime-transition results on modern 4-bit instruct models.}",
            "\\label{tab:main-modern-results}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Model & Context & Online & KVRegime & Oracle \\\\",
            "\\midrule",
            "Qwen2.5-1.5B & 256 & 1.000 & 1.000 & 1.274 \\\\",
            "Qwen2.5-1.5B & 512 & 1.169 & 1.169 & 1.306 \\\\",
            "Qwen3-8B & 256 & 1.155 & 1.155 & 1.277 \\\\",
            "SmolLM2-1.7B & 256 & 1.150 & 1.150 & 1.276 \\\\",
            "Phi-3.5-mini & 256 & 1.140 & 1.140 & 1.276 \\\\",
            "Phi-3.5-mini & 384 & 1.181 & 1.181 & 1.297 \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    (TABLE_DIR / "main_results_table.tex").write_text(main_table + "\n", encoding="utf-8")

    onset_table = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Family-dependent onset of the positive offload regime.}",
            "\\label{tab:family-onset}",
            "\\begin{tabular}{lc}",
            "\\toprule",
            "Family & First online-positive context \\\\",
            "\\midrule",
            "gpt2 & 512 \\\\",
            "opt-125m & 512 \\\\",
            "Qwen2.5-1.5B & 384 \\\\",
            "Qwen3-8B & 256 \\\\",
            "SmolLM2-1.7B & 256 \\\\",
            "Phi-3.5-mini & 256 \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    (TABLE_DIR / "family_onset_table.tex").write_text(onset_table + "\n", encoding="utf-8")

    policy = practical_policy_payload()["summary"]
    policy_order = [
        ("gpu_only", "GPU-only"),
        ("static_attention", "Static attention"),
        ("context_threshold", "Context threshold"),
        ("adaptive_feature", "Adaptive feature"),
        ("kv_regime", "KV regime"),
        ("context_kv_refined", "Context+KV"),
        ("context_kv_hysteresis", "Context+KV+hyst."),
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Practical policy trade-offs over the 34-workload evaluation set. The added context+KV ablation isolates hysteresis itself: under the current trace granularity, hysteresis does not materially change aggregate behavior, so the practical gain comes primarily from conservative context+KV gating rather than from hysteresis alone.}",
        "\\label{tab:practical-policy}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Policy & Speedup & Oracle Regret & FP Rate & Missed-Pos. & Switches /100 \\\\",
        "\\midrule",
    ]
    for key, label in policy_order:
        row = policy[key]
        lines.append(
            f"{label} & {row['speedup_vs_gpu_only']:.3f} & {row['oracle_regret']:.3f} & {row['false_positive_rate']:.3f} & {row['missed_positive_rate']:.3f} & {row['switches_per_100']:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (TABLE_DIR / "practical_policy_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_assets() -> None:
    (FINAL_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Final Paper Package",
                "",
                "This folder contains the paper-ready assets derived from the current experiment set.",
                "",
                "## Figures",
                "",
                "- `figures/paper_backbone_context_transition.pdf`",
                "- `figures/paper_backbone_kv_pressure_vs_speedup.pdf`",
                "- `figures/paper_policy_comparison.pdf`",
                "- `figures/paper_negative_cases.pdf`",
                "- `figures/paper_decision_signal_study.pdf`",
                "- `figures/paper_same_context_family.pdf`",
                "- `figures/paper_backend_validation.pdf`",
                "- `figures/paper_family_onset_summary.pdf`",
                "- `figures/paper_sensitivity_summary.pdf`",
                "",
                "## Tables",
                "",
                "- `tables/main_results_table.tex`",
                "- `tables/family_onset_table.tex`",
                "- `tables/backend_measured_vs_modeled.tex`",
                "- `tables/backend_parameters.tex`",
                "- `tables/practical_policy_table.tex`",
                "",
                "## Writing Assets",
                "",
                "- `figure_captions.md`",
                "- `negative_results.md`",
                "- `policy_story.md`",
                "- `paper_backbone.md`",
                "- `results_insert.md`",
                "- `latex_snippets.tex`",
                "- `asset_index.md`",
                "- `backend_formalization.md`",
                "",
                "## Framework Diagram",
                "",
                "- `framework_overview.mmd`",
                "- `framework_overview.md`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (FINAL_DIR / "figure_captions.md").write_text(
        "\n".join(
            [
                "# Figure Captions",
                "",
                "## Figure 1: Context-Driven Regime Transition",
                "",
                "Dense context sweeps across four modern instruct families show that virtual PIM benefit is not universal but emerges after a family-dependent context threshold. `Qwen2.5-1.5B` shows delayed onset (`256 -> 384`), while `Qwen3-8B`, `SmolLM2-1.7B`, and `Phi-3.5-mini` enter the positive regime earlier. In our current setup, context length is the clearest coarse predictor of regime entry, while model family shifts the onset boundary.",
                "",
                "## Figure 2: KV-Related Pressure Is Consistent With Positive Offload Regimes",
                "",
                "Across dense context sweeps and the extra `Phi-3.5-mini` validation, higher memory-pressure proxy values tend to align with positive online speedup. This figure provides predictive evidence that the observed regime boundary is consistent with rising KV-cache-related pressure, without claiming a fully causal mechanism.",
                "",
                "## Figure 3: Static and Generic Realistic Policies Miss Early-Positive Cases",
                "",
                "On selected delayed-onset and early-onset modern-model cases, `static_attention` is directionally useful but insufficient, while `kv_regime` sometimes recovers early-positive families that `adaptive_feature` misses at short context. This figure should be read as a policy case study rather than as a universal scheduler ranking.",
                "",
                "## Figure 4: Negative Results as a Distribution, Not a Case Study",
                "",
                "The left panel shows the workload-level distribution of realistic online gains, while the right panel shows positive-regime fraction versus context. Together they show that gains are not fabricated uniformly: short-context workloads frequently remain weak, while mid- and long-context workloads are much more often positive.",
                "",
                "## Figure 5: Held-Out Decision-Signal Study",
                "",
                "Held-out workload classification over the same 34 workloads uses backend-defined positive-regime labels from the current virtual replay setup. Under that fixed definition, attention-only signals over-predict positive regimes, while context length is the strongest coarse predictor. Adding KV-related features modestly improves context-bucket-held-out detection, which supports using them as complementary decision signals rather than as standalone causal proof.",
                "",
                "## Figure 6: Same-Context Cross-Family Comparison",
                "",
                "At a fixed `ctx=256`, delayed-onset `Qwen2.5` remains weak while early-positive `Qwen3`, `SmolLM2`, and `Phi-3.5` are already positive. The stronger cases also carry larger KV-byte scale at the same context. This gives a more concrete family-level readout than a context-only narrative, while still remaining predictive rather than causal evidence.",
                "",
                "## Figure 7: Backend Validation Appendix Figure",
                "",
                "The left panel shows preset-level positive fractions across conservative/default/optimistic backends, while the right panel shows first-positive context for representative families across the same presets. This makes the sensitivity argument concrete: preset changes shift absolute positivity rates, but delayed-onset families remain delayed and early-positive families remain early-positive.",
                "",
                "## Figure 8: Family-Dependent Onset Summary",
                "",
                "The first positive context varies by family, with `gpt2` and `opt-125m` entering later than `Qwen3`, `SmolLM2`, and `Phi-3.5`. This compact view summarizes the paper’s family-dependent onset claim.",
                "",
                "## Figure 9: Cost-Model Robustness Summary",
                "",
                "Conservative/default/optimistic virtual-PIM presets change absolute speedups but preserve the main qualitative conclusion: canonical long-context positive cases remain positive, while weak or borderline short-context cases stay near the boundary.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (FINAL_DIR / "negative_results.md").write_text(
        "\n".join(
            [
                "# Negative Results Subsection",
                "",
                "A key result of this study is that virtual PIM value is not automatic. The workload-level gain distribution shows a clear mass near `1.0x` for short-context runs even though oracle-only upside may still exist. The positive-regime fraction curve makes the same point more cleanly: only `4/11` short-context workloads are online-positive, while the fraction rises to `9/9` for mid-context workloads and `14/14` for long-context workloads.",
                "",
                "These negative results are central to the paper because they improve credibility. Rather than selecting a few favorable cases, the distributional view shows that weak regimes are common and structured. This supports a narrower claim: virtual PIM offload becomes interesting only after the decode stream enters a favorable regime, and short-context execution often fails to provide that signal.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (FINAL_DIR / "policy_story.md").write_text(
        "\n".join(
            [
                "# Policy Story Subsection",
                "",
                "The policy results suggest that attention labeling alone is not the right abstraction for realistic scheduling. A static attention heuristic is directionally useful, but it is too permissive: in the held-out decision-signal study, attention-only cues achieve high positive recall while also incurring a high false-positive rate. The core decision problem is therefore not simply to detect attention regions, but to decide whether a decode stream has plausibly crossed into a positive offload regime under the current replay model.",
                "",
                "The stronger claim supported by the current data is more modest than a new scheduler result. Context length is the clearest coarse predictor in our current setup, while KV-related signals help refine boundary detection when coarse context cues are withheld. The case-study comparison between `adaptive_feature` and `kv_regime` is therefore best framed as an existence proof that explicit KV-aware features can matter on some early-positive families, not as proof that one hand-written policy universally dominates another.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (FINAL_DIR / "paper_backbone.md").write_text(
        "\n".join(
            [
                "# Paper Backbone",
                "",
                "## Core Message",
                "",
                "Virtual PIM value during LLM decoding is regime-dependent. The clearest coarse separator in the current setup is context-sensitive regime entry, and the observed transition is consistent with rising KV-cache-related pressure. The exact onset is family-dependent, which is why static attention rules and generic realistic heuristics can mis-handle some early-positive modern-model cases.",
                "",
                "## Backbone Flow",
                "",
                "1. Start with the negative result: short context does not automatically benefit from offload.",
                "2. Show the context-transition figure: context governs regime entry, but family shifts the onset.",
                "3. Show the KV-pressure figure: the regime shift is consistent with rising KV-cache-driven pressure.",
                "4. Show the distributional negative-result figure: gains do not appear automatically.",
                "5. Show the held-out decision-signal figure: attention-only over-predicts, context is the clearest coarse signal, and KV-related features are complementary.",
                "6. Close with the case-study policy comparison plus robustness and batch-control summaries as defense rather than as the main message.",
                "",
                "## Contribution Wording",
                "",
                "- We provide a trace-driven framework for evaluating virtual PIM scheduling during LLM decoding.",
                "- We show that offload value appears in context-sensitive positive regimes rather than uniformly across workloads.",
                "- We show that context length is the clearest coarse predictor of regime entry in our current setup, and that onset varies across model families.",
                "- We provide evidence that the observed boundary is consistent with KV-cache-related pressure and that KV-aware features can complement coarse context cues for decision support.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (FINAL_DIR / "results_insert.md").write_text(
        "\n".join(
            [
                "# Results Insert",
                "",
                "Our results show a consistent weak-signal versus positive-signal split. Short-context decoding does not automatically benefit from virtual PIM offload, which appears clearly in the workload-level gain distribution and the context-wise positive-fraction curves. In contrast, denser context sweeps reveal a family-dependent regime transition: `Qwen2.5-1.5B` transitions later (`256 -> 384`), while `Qwen3-8B`, `SmolLM2-1.7B`, and `Phi-3.5-mini` enter the positive regime earlier. This makes context length the clearest coarse regime-entry axis in our study, while also showing that onset is not universal across families.",
                "",
                "The mechanism evidence is predictive rather than causal. Across focused GPU studies, higher memory-pressure proxy values align with positive online speedup more directly than attention share alone, and held-out workload classification shows that attention-only cues are too permissive while all-feature detectors modestly improve boundary detection when coarse context cues are withheld. Taken together, these results support a narrower claim than a static attention-offload story: virtual PIM value emerges in context-sensitive regimes, the observed boundary is consistent with KV-cache-related pressure, and realistic decision support should not rely on attention labels alone.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (FINAL_DIR / "asset_index.md").write_text(
        "\n".join(
            [
                "# Asset Index",
                "",
                "| Asset | Role In Paper |",
                "|---|---|",
                "| `figures/paper_backbone_context_transition.pdf` | Main backbone figure for regime entry |",
                "| `figures/paper_backbone_kv_pressure_vs_speedup.pdf` | Main backbone figure for mechanism evidence |",
                "| `figures/paper_policy_comparison.pdf` | Policy contribution figure |",
                "| `figures/paper_negative_cases.pdf` | Trust-building negative-result figure |",
                "| `figures/paper_decision_signal_study.pdf` | Held-out signal-ablation figure |",
                "| `figures/paper_same_context_family.pdf` | Fixed-context cross-family mechanism support figure |",
                "| `figures/paper_backend_validation.pdf` | Appendix robustness figure for backend presets and threshold sensitivity |",
                "| `figures/paper_family_onset_summary.pdf` | Compact family-onset summary |",
                "| `figures/paper_sensitivity_summary.pdf` | Robustness defense figure |",
                "| `tables/main_results_table.tex` | Main results table |",
                "| `tables/family_onset_table.tex` | Family-dependent onset table |",
                "| `tables/backend_measured_vs_modeled.tex` | Backend provenance split table |",
                "| `tables/backend_parameters.tex` | Backend parameter table |",
                "| `tables/practical_policy_table.tex` | Practical-policy trade-off table |",
                "| `negative_results.md` | Results subsection draft |",
                "| `policy_story.md` | Policy subsection draft |",
                "| `paper_backbone.md` | Section ordering and message spine |",
                "| `backend_formalization.md` | Backend equations and interpretation boundary |",
                "| `latex_snippets.tex` | Direct LaTeX include snippets |",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (FINAL_DIR / "backend_formalization.md").write_text(
        (ROOT / "artifacts/reports/BACKEND_FORMALIZATION.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    (FINAL_DIR / "latex_snippets.tex").write_text(
        "\n".join(
            [
                "% Final paper package snippets",
                "\\begin{figure}[t]",
                "    \\centering",
                "    \\includegraphics[width=0.95\\linewidth]{final/figures/paper_backbone_context_transition.pdf}",
                "    \\caption{Context-driven regime transition across modern instruct families.}",
                "    \\label{fig:context-transition}",
                "\\end{figure}",
                "",
                "\\begin{figure}[t]",
                "    \\centering",
                "    \\includegraphics[width=0.95\\linewidth]{final/figures/paper_backbone_kv_pressure_vs_speedup.pdf}",
                "    \\caption{KV-related memory pressure tracks positive offload regimes.}",
                "    \\label{fig:kv-pressure}",
                "\\end{figure}",
                "",
                "\\begin{figure}[t]",
                "    \\centering",
                "    \\includegraphics[width=0.95\\linewidth]{final/figures/paper_policy_comparison.pdf}",
                "    \\caption{Static and generic realistic policies miss some early-positive modern-family cases, while KV-aware detection recovers them.}",
                "    \\label{fig:policy-comparison}",
                "\\end{figure}",
                "",
                "\\begin{figure}[t]",
                "    \\centering",
                "    \\includegraphics[width=0.95\\linewidth]{final/figures/paper_negative_cases.pdf}",
                "    \\caption{Distributional negative-result evidence.}",
                "    \\label{fig:negative-cases}",
                "\\end{figure}",
                "",
                "\\begin{figure}[t]",
                "    \\centering",
                "    \\includegraphics[width=0.95\\linewidth]{final/figures/paper_decision_signal_study.pdf}",
                "    \\caption{Held-out decision-signal comparison for workload-level positive-regime detection under the fixed virtual backend definition.}",
                "    \\label{fig:decision-signal-study}",
                "\\end{figure}",
                "",
                "\\begin{figure}[t]",
                "    \\centering",
                "    \\includegraphics[width=0.95\\linewidth]{final/figures/paper_same_context_family.pdf}",
                "    \\caption{Fixed-context cross-family comparison at $\\ctx=256$ showing that delayed-onset and early-positive families can differ in KV-byte scale and online speedup even at the same prompt length.}",
                "    \\label{fig:same-context-family}",
                "\\end{figure}",
                "",
                "\\begin{figure}[t]",
                "    \\centering",
                "    \\includegraphics[width=0.95\\linewidth]{final/figures/paper_backend_validation.pdf}",
                "    \\caption{Appendix robustness view: backend preset stability and label-threshold sensitivity.}",
                "    \\label{fig:backend-validation}",
                "\\end{figure}",
                "",
                "\\input{final/tables/main_results_table.tex}",
                "\\input{final/tables/family_onset_table.tex}",
                "\\input{final/tables/backend_measured_vs_modeled.tex}",
                "\\input{final/tables/backend_parameters.tex}",
                "\\input{final/tables/practical_policy_table.tex}",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    generate_context_transition()
    generate_kv_pressure_scatter()
    generate_policy_comparison()
    generate_negative_cases()
    generate_decision_signal_study()
    generate_same_context_family()
    generate_backend_validation_figure()
    generate_family_onset_summary()
    generate_sensitivity_summary()
    build_framework_mermaid()
    write_tables()
    write_markdown_assets()
    print(f"wrote final package to {FINAL_DIR}")


if __name__ == "__main__":
    main()
