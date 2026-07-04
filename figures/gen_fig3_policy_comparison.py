from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, save_fig


ROOT = Path(__file__).resolve().parents[1]


CASES = [
    ("GPT2-256", ROOT / "artifacts/reports/gpt2_gpu_ctx256_bs1_simulation_full.json"),
    ("GPT2-768", ROOT / "artifacts/reports/gpt2_gpu_ctx768_bs1_simulation_full.json"),
    ("OPT-512", ROOT / "artifacts/reports/opt125m_gpu_ctx512_bs1_simulation_full.json"),
    ("Qwen2.5-512", ROOT / "artifacts/reports/qwen_qwen2_5_1_5b_instruct_4bit_ctx512_bs1_latest_probe_simulation_full.json"),
    ("Qwen3-512", ROOT / "artifacts/reports/unsloth_qwen3_8b_unsloth_bnb_4bit_none_ctx512_bs1_qwen3_unsloth_probe_simulation.json"),
    ("SmolLM2-256", ROOT / "artifacts/reports/huggingfacetb_smollm2_1_7b_instruct_4bit_ctx256_bs1_latest_probe_simulation_full.json"),
]

POLICIES = ["gpu_only", "static_attention", "online_predictor", "adaptive_feature", "oracle"]


def load_speeds(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["policy_name"]: row["speedup_vs_gpu_only"] for row in payload["results"]}


def main() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 3.8))

    x = np.arange(len(CASES))
    width = 0.15
    offsets = np.linspace(-2, 2, len(POLICIES)) * width

    for policy_idx, policy in enumerate(POLICIES):
        vals = [load_speeds(path)[policy] for _, path in CASES]
        ax.bar(x + offsets[policy_idx], vals, width=width, label=policy, color=COLORS[policy_idx % len(COLORS)])

    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in CASES], rotation=20, ha="right")
    ax.set_ylabel("Speedup vs GPU-Only")
    ax.set_ylim(0.95, 1.36)
    ax.legend(frameon=False, ncol=3)
    save_fig(fig, "fig3_policy_comparison")


if __name__ == "__main__":
    main()
