from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from paper_plot_style import COLORS, FIG_DIR, save_fig


ROOT = FIG_DIR.parent
INPUT = ROOT / "artifacts" / "reports" / "qwen2_5_controlled_batch.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    payload = load(INPUT)
    rows = sorted(payload["rows"], key=lambda item: (int(item["context"]), int(item["batch_size"])))
    contexts = sorted({int(row["context"]) for row in rows})

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), sharex=True)
    for idx, context in enumerate(contexts):
        ax = axes[idx]
        ctx_rows = [row for row in rows if int(row["context"]) == context]
        batches = [int(row["batch_size"]) for row in ctx_rows]
        speedup = [float(row["online_speedup"]) for row in ctx_rows]
        kv_mb = [float(row["mean_kv_bytes_est"]) / (1024.0 * 1024.0) for row in ctx_rows]

        ax2 = ax.twinx()
        ax.plot(batches, speedup, marker="o", color=COLORS[0], linewidth=1.8)
        ax2.plot(batches, kv_mb, marker="s", color=COLORS[1], linewidth=1.8)
        ax.axhline(1.05, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(f"Context {context}")
        ax.set_xlabel("Batch Size")
        ax.set_xticks(batches)
        if idx == 0:
            ax.set_ylabel("Online Speedup")
        if idx == len(axes) - 1:
            ax2.set_ylabel("KV Bytes (MB)")

    fig.suptitle("Batch Scales KV Traffic but Does Not Shift the Context Regime", y=1.03)
    save_fig(fig, "fig10_batch_context_control")


if __name__ == "__main__":
    main()
