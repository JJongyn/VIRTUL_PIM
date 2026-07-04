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
    selected_models = [
        "gpt2",
        "opt-125m",
        "qwen2.5-1.5b-instruct-4bit",
        "qwen3-8b-4bit",
        "smollm2-1.7b-instruct-4bit",
    ]

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.8))
    color_idx = 0
    for model in selected_models:
        model_rows = [
            r for r in rows
            if r["device_kind"] == "gpu"
            and r["model"] == model
            and int(float(r["batch_size"])) == 1
        ]
        if len(model_rows) < 2:
            continue
        model_rows.sort(key=lambda r: int(float(r["context"])))
        xs = [int(float(r["context"])) for r in model_rows]
        ys = [float(r["online_speedup"]) for r in model_rows]
        label = model.replace("-4bit", "").replace("-instruct", "")
        ax.plot(xs, ys, marker="o", linewidth=2, color=COLORS[color_idx % len(COLORS)], label=label)
        color_idx += 1

    ax.set_xlabel("Context Length")
    ax.set_ylabel("Online Speedup vs GPU-Only")
    ax.set_ylim(0.95, 1.35)
    ax.legend(frameon=False, ncol=2)
    save_fig(fig, "fig1_context_vs_speedup")


if __name__ == "__main__":
    main()
