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
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=False)
    for idx, (ax, path) in enumerate(zip(axes, INPUTS)):
        if not path.exists():
            continue
        payload = load(path)
        rows = sorted(payload["rows"], key=lambda item: int(item["context"]))
        xs = [int(row["context"]) for row in rows]
        kv_mb = [float(row["mean_kv_bytes_est"]) / (1024.0 * 1024.0) for row in rows]
        speedup = [float(row["online_speedup"]) for row in rows]

        ax2 = ax.twinx()
        ax.plot(xs, kv_mb, marker="o", color=COLORS[0], linewidth=1.8)
        ax2.plot(xs, speedup, marker="s", color=COLORS[1], linewidth=1.8)
        ax.set_title(payload["model"])
        ax.set_xlabel("Context")
        if idx == 0:
            ax.set_ylabel("KV Bytes (MB)")
        ax2.set_ylim(bottom=0.95)
        if idx == len(axes) - 1:
            ax2.set_ylabel("Online Speedup")

    fig.suptitle("Context Growth Increases KV Traffic and Offload Value", y=1.03)
    save_fig(fig, "fig8_context_kv_speedup_dualaxis")


if __name__ == "__main__":
    main()
