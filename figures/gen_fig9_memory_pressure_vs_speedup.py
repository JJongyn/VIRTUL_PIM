from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from paper_plot_style import COLORS, FIG_DIR, save_fig


ROOT = FIG_DIR.parent
INPUTS = [
    ROOT / "artifacts" / "reports" / "qwen2_5_dense_context_sweep.json",
    ROOT / "artifacts" / "reports" / "qwen3_dense_context_sweep.json",
    ROOT / "artifacts" / "reports" / "smollm2_dense_context_sweep.json",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for idx, path in enumerate(INPUTS):
        if not path.exists():
            continue
        payload = load(path)
        rows = sorted(payload["rows"], key=lambda item: int(item["context"]))
        xs = [float(row["mean_memory_pressure_proxy"]) for row in rows]
        ys = [float(row["online_speedup"]) for row in rows]
        sizes = [28 + int(row["context"]) / 24 for row in rows]
        ax.scatter(xs, ys, s=sizes, color=COLORS[idx % len(COLORS)], alpha=0.85, label=payload["model"])
        for row in rows:
            ax.annotate(str(int(row["context"])), (float(row["mean_memory_pressure_proxy"]), float(row["online_speedup"])), fontsize=7, xytext=(3, 2), textcoords="offset points")

    ax.axhline(1.05, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Memory Pressure Proxy")
    ax.set_ylabel("Online Speedup vs GPU-Only")
    ax.set_title("Memory Pressure Tracks Positive Offload Regimes")
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, "fig9_memory_pressure_vs_speedup")


if __name__ == "__main__":
    main()
