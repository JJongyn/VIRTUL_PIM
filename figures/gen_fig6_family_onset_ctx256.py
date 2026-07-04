from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, FIG_DIR, save_fig


DATA = FIG_DIR.parent / "artifacts" / "reports" / "kv_mechanism_dataset.csv"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = [
        row for row in load_rows(DATA)
        if int(row["context"]) == 256 and int(row["batch_size"]) == 1
    ]
    rows = sorted(rows, key=lambda item: item["model"])

    labels = [row["model"] for row in rows]
    online = [float(row["online_speedup"]) for row in rows]
    adaptive = [float(row["adaptive_feature_speedup"]) for row in rows]
    kv_regime = [float(row["kv_regime_speedup"]) for row in rows]

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.bar(x - width, online, width=width, color=COLORS[0], label="Online")
    ax.bar(x, adaptive, width=width, color=COLORS[1], label="AdaptiveFeature")
    ax.bar(x + width, kv_regime, width=width, color=COLORS[2], label="KVRegime")
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Speedup vs GPU-Only")
    ax.set_title("Family-Dependent Early Onset at Context 256")
    ax.legend(frameon=False, ncol=3)
    save_fig(fig, "fig6_family_onset_ctx256")


if __name__ == "__main__":
    main()
