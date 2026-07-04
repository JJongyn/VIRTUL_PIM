from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from paper_plot_style import COLORS, save_fig


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "artifacts" / "reports" / "full_correlation_dataset.csv"


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = load_rows()
    families = {
        "legacy": ["gpt2", "opt-125m"],
        "modern_qwen": ["qwen2.5-1.5b-instruct-4bit", "qwen3-8b-4bit"],
        "modern_other": ["smollm2-1.7b-instruct-4bit"],
    }
    markers = {"legacy": "o", "modern_qwen": "s", "modern_other": "^"}

    fig, ax = plt.subplots(1, 1, figsize=(5.8, 3.8))
    for idx, (group, models) in enumerate(families.items()):
        group_rows = [r for r in rows if r["device_kind"] == "gpu" and r["model"] in models]
        xs = [float(r["attention_share"]) for r in group_rows]
        ys = [float(r["online_speedup"]) for r in group_rows]
        ax.scatter(xs, ys, s=40, alpha=0.85, marker=markers[group], color=COLORS[idx], label=group.replace("_", " "))

    ax.set_xlabel("Attention Latency Share")
    ax.set_ylabel("Online Speedup vs GPU-Only")
    ax.set_xlim(0.40, 0.90)
    ax.set_ylim(0.95, 1.35)
    ax.legend(frameon=False)
    save_fig(fig, "fig2_attention_vs_speedup")


if __name__ == "__main__":
    main()
