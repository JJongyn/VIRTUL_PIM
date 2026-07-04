from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from paper_plot_style import COLORS, FIG_DIR, save_fig


DATA = FIG_DIR.parent / "artifacts" / "reports" / "kv_mechanism_dataset.csv"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = load_rows(DATA)
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["batch_size"]) != 1:
            continue
        by_model[row["model"]].append(row)

    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for idx, (model, model_rows) in enumerate(sorted(by_model.items())):
        model_rows = sorted(model_rows, key=lambda item: int(item["context"]))
        xs = [int(row["context"]) for row in model_rows]
        ys = [float(row["mean_kv_bytes_est"]) / (1024.0 * 1024.0) for row in model_rows]
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=COLORS[idx % len(COLORS)], label=model)

    ax.set_xlabel("Prompt Context Length")
    ax.set_ylabel("Estimated KV Bytes (MB)")
    ax.set_title("Context Length Tracks KV-Cache Traffic")
    ax.legend(frameon=False, ncol=1)
    save_fig(fig, "fig4_context_vs_kvbytes")


if __name__ == "__main__":
    main()
