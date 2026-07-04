from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from paper_plot_style import COLORS, FIG_DIR, save_fig


DATA = FIG_DIR.parent / "artifacts" / "reports" / "kv_mechanism_dataset.csv"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = load_rows(DATA)
    fig, ax = plt.subplots(figsize=(4.4, 3.1))

    model_to_color: dict[str, tuple[float, float, float, float]] = {}
    for idx, model in enumerate(sorted({row["model"] for row in rows})):
        model_to_color[model] = COLORS[idx % len(COLORS)]

    for row in rows:
        x = float(row["mean_kv_bytes_est"]) / (1024.0 * 1024.0)
        y = float(row["online_speedup"])
        color = model_to_color[row["model"]]
        marker = "s" if int(row["batch_size"]) > 1 else "o"
        ax.scatter(x, y, color=color, marker=marker, s=40, alpha=0.9)

    for model, color in model_to_color.items():
        ax.scatter([], [], color=color, marker="o", s=40, label=model)

    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Estimated KV Bytes (MB)")
    ax.set_ylabel("Online Speedup vs GPU-Only")
    ax.set_title("Higher KV Traffic Aligns with Higher Offload Value")
    ax.legend(frameon=False, ncol=1)
    save_fig(fig, "fig5_kvbytes_vs_speedup")


if __name__ == "__main__":
    main()
