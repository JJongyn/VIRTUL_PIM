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
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for idx, path in enumerate(INPUTS):
        if not path.exists():
            continue
        payload = load(path)
        rows = sorted(payload["rows"], key=lambda item: int(item["context"]))
        xs = [int(row["context"]) for row in rows]
        ys = [float(row["online_speedup"]) for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=COLORS[idx % len(COLORS)], label=payload["model"])

    ax.axhline(1.05, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Prompt Context Length")
    ax.set_ylabel("Online Speedup vs GPU-Only")
    ax.set_title("Dense Context Sweeps Reveal Regime Transition")
    ax.legend(frameon=False)
    save_fig(fig, "fig7_dense_context_transition")


if __name__ == "__main__":
    main()
