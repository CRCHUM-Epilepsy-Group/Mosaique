#!/usr/bin/env python
"""Plot benchmark results: RAM and wall time vs number of files.

Usage
-----
    uv run benchmark/plot.py
    uv run benchmark/plot.py --input benchmark/output/benchmark.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"

FEATURE_GROUPS = ["simple", "tf_decomposition", "connectivity"]
GROUP_LABELS = {
    "simple": "Simple",
    "tf_decomposition": "TF Decomposition",
    "connectivity": "Connectivity",
}

# Colors: MNE gets one color, Mosaïque gets a color per worker count
MNE_COLOR = "#e05c5c"
MOSAIQUE_COLORS = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]


def load_medians(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    return (
        df.group_by(["backend", "n_files", "feature_group", "n_workers"])
        .agg(
            pl.col("wall_s").median(),
            pl.col("peak_rss_mb").median(),
        )
        .sort(["feature_group", "backend", "n_workers", "n_files"])
    )


def plot_metric(
    df: pl.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    worker_counts = sorted(
        df.filter(pl.col("backend") == "mosaique")["n_workers"].unique().to_list()
    )

    for ax, group in zip(axes, FEATURE_GROUPS):
        gdf = df.filter(pl.col("feature_group") == group)

        # MNE line (always 1 worker)
        mne = gdf.filter(pl.col("backend") == "mne").sort("n_files")
        ax.plot(
            mne["n_files"].to_list(),
            mne[metric].to_list(),
            marker="o",
            color=MNE_COLOR,
            label="MNE",
            linewidth=2,
        )

        # Mosaïque lines per worker count
        for i, w in enumerate(worker_counts):
            mos = gdf.filter(
                (pl.col("backend") == "mosaique") & (pl.col("n_workers") == w)
            ).sort("n_files")
            ax.plot(
                mos["n_files"].to_list(),
                mos[metric].to_list(),
                marker="s",
                color=MOSAIQUE_COLORS[i % len(MOSAIQUE_COLORS)],
                label=f"Mosaïque w={w}",
                linewidth=2,
            )

        ax.set_title(GROUP_LABELS[group])
        ax.set_xlabel("Number of files")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(df["n_files"].unique().to_list()))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot benchmark results")
    p.add_argument(
        "--input",
        type=Path,
        default=OUTPUT_DIR / "benchmark.parquet",
        help="Path to benchmark parquet file",
    )
    args = p.parse_args()

    df = load_medians(args.input)

    plot_metric(
        df,
        metric="wall_s",
        ylabel="Wall time (s)",
        title="Wall time vs number of files",
        output_path=OUTPUT_DIR / "wall_time.png",
    )

    plot_metric(
        df,
        metric="peak_rss_mb",
        ylabel="Peak RAM (MB)",
        title="Peak RAM usage vs number of files",
        output_path=OUTPUT_DIR / "ram_usage.png",
    )


if __name__ == "__main__":
    main()
