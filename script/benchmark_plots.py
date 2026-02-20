#!/usr/bin/env python
"""Generate publication-ready benchmark figures from Parquet results.

Reads the output of ``benchmark.py`` and produces five plot types:

1. Speedup heatmap (per feature group)
2. Wall time vs N files
3. Worker scaling (bar chart)
4. Feature scaling (line plot)
5. Memory comparison (bar chart)

Usage
-----
    uv run script/benchmark_plots.py
    uv run script/benchmark_plots.py --input path/to/benchmark.parquet
    uv run script/benchmark_plots.py --output-dir path/to/plots/

Outputs PNG files (300 dpi).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "output" / "benchmark.parquet"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "plots"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

COLOR_MNE = "#d95f02"
COLOR_MOS = {1: "#7fc7ff", 2: "#3498db", 4: "#1a5276"}
COLOR_MOS_DEFAULT = "#2980b9"

LABEL_SIZE = 12
TICK_SIZE = 10
TITLE_SIZE = 13

GROUP_LABELS = {
    "simple": "Simple",
    "tf_decomposition": "TF Decomposition",
    "connectivity": "Connectivity",
}


def _footnote(ax: Any, df: pl.DataFrame) -> None:
    """Add system metadata as a small footnote below the plot."""
    row = df.row(0, named=True)
    parts = [
        row.get("cpu_model", ""),
        f"{row.get('n_cores', '?')} cores",
        f"{row.get('ram_gb', '?')} GB RAM",
        f"git {row.get('git_commit', '?')}",
    ]
    text = " | ".join(p for p in parts if p)
    ax.annotate(
        text,
        xy=(0.5, -0.02),
        xycoords="figure fraction",
        ha="center",
        fontsize=7,
        color="grey",
    )


# ---------------------------------------------------------------------------
# 1. Speedup Heatmap
# ---------------------------------------------------------------------------

def plot_speedup_heatmap(df: pl.DataFrame, output_dir: Path) -> None:
    """One heatmap per feature group: rows=n_files, cols=n_features."""
    groups: list[str] = df["feature_group"].unique().drop_nulls().sort().to_list()
    n_groups = len(groups)

    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4.5), squeeze=False)

    for idx, group in enumerate(groups):
        ax = axes[0, idx]
        gdf = df.filter(pl.col("feature_group") == group)

        mne_med = (
            gdf.filter(pl.col("backend") == "mne")
            .group_by(["n_files", "n_features"])
            .agg(pl.col("wall_s").median().alias("mne_wall"))
        )
        mos_med = (
            gdf.filter(pl.col("backend") == "mosaique")
            .group_by(["n_files", "n_features"])
            .agg(pl.col("wall_s").median().alias("mos_wall"))
            .sort("mos_wall")
            .group_by(["n_files", "n_features"])
            .first()
        )

        merged = mne_med.join(mos_med, on=["n_files", "n_features"])
        merged = merged.with_columns(
            (pl.col("mne_wall") / pl.col("mos_wall")).alias("speedup")
        )

        n_files_vals = sorted(merged["n_files"].unique().to_list())
        n_feat_vals = sorted(merged["n_features"].unique().to_list())

        grid = np.full((len(n_files_vals), len(n_feat_vals)), np.nan)
        for row in merged.iter_rows(named=True):
            r = n_files_vals.index(row["n_files"])
            c = n_feat_vals.index(row["n_features"])
            grid[r, c] = row["speedup"]

        vmax = max(abs(np.nanmax(grid)), abs(1 / np.nanmin(grid))) if not np.all(np.isnan(grid)) else 2
        im = ax.imshow(
            grid,
            aspect="auto",
            cmap="RdYlGn",
            vmin=1 / vmax,
            vmax=vmax,
            origin="lower",
        )

        # Annotate cells
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                val = grid[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.2f}x", ha="center", va="center", fontsize=9)

        ax.set_xticks(range(len(n_feat_vals)))
        ax.set_xticklabels(n_feat_vals, fontsize=TICK_SIZE)
        ax.set_yticks(range(len(n_files_vals)))
        ax.set_yticklabels(n_files_vals, fontsize=TICK_SIZE)
        ax.set_xlabel("N features", fontsize=LABEL_SIZE)
        ax.set_ylabel("N files", fontsize=LABEL_SIZE)
        ax.set_title(GROUP_LABELS.get(group, group), fontsize=TITLE_SIZE)

        fig.colorbar(im, ax=ax, label="Speedup (Mosaique vs MNE)")

    fig.suptitle("Speedup Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _footnote(axes[0, 0], df)
    fig.savefig(output_dir / "speedup_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Wall Time vs N Files
# ---------------------------------------------------------------------------

def plot_wall_vs_nfiles(df: pl.DataFrame, output_dir: Path) -> None:
    """Line plot per group: MNE vs Mosaique at various worker counts."""
    groups: list[str] = df["feature_group"].unique().drop_nulls().sort().to_list()
    n_groups = len(groups)

    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4), squeeze=False)

    for idx, group in enumerate(groups):
        ax = axes[0, idx]
        gdf = df.filter(pl.col("feature_group") == group)

        # Use max features
        max_nf = gdf["n_features"].max()
        gdf = gdf.filter(pl.col("n_features") == max_nf)

        # MNE line
        mne_stats = (
            gdf.filter(pl.col("backend") == "mne")
            .group_by("n_files")
            .agg(
                pl.col("wall_s").median().alias("median"),
                pl.col("wall_s").std().alias("std"),
            )
            .sort("n_files")
        )
        ax.errorbar(
            mne_stats["n_files"].to_list(),
            mne_stats["median"].to_list(),
            yerr=mne_stats["std"].fill_null(0).to_list(),
            label="MNE",
            color=COLOR_MNE,
            linestyle="--",
            marker="o",
            capsize=3,
        )

        # Mosaique lines per worker count
        mos_df = gdf.filter(pl.col("backend") == "mosaique")
        workers = sorted(mos_df["n_workers"].unique().to_list())
        for w in workers:
            wdf = (
                mos_df.filter(pl.col("n_workers") == w)
                .group_by("n_files")
                .agg(
                    pl.col("wall_s").median().alias("median"),
                    pl.col("wall_s").std().alias("std"),
                )
                .sort("n_files")
            )
            ax.errorbar(
                wdf["n_files"].to_list(),
                wdf["median"].to_list(),
                yerr=wdf["std"].fill_null(0).to_list(),
                label=f"Mosaique {w}w",
                color=COLOR_MOS.get(w, COLOR_MOS_DEFAULT),
                marker="s",
                capsize=3,
            )

        ax.set_xlabel("N files", fontsize=LABEL_SIZE)
        ax.set_ylabel("Wall time (s)", fontsize=LABEL_SIZE)
        ax.set_title(GROUP_LABELS.get(group, group), fontsize=TITLE_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.legend(fontsize=9)

    fig.suptitle("Wall Time vs N Files", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _footnote(axes[0, 0], df)
    fig.savefig(output_dir / "wall_vs_nfiles.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Worker Scaling
# ---------------------------------------------------------------------------

def plot_worker_scaling(df: pl.DataFrame, output_dir: Path) -> None:
    """Bar chart per group: wall time by n_workers, grouped by n_files."""
    groups: list[str] = df["feature_group"].unique().drop_nulls().sort().to_list()
    n_groups = len(groups)

    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4), squeeze=False)

    for idx, group in enumerate(groups):
        ax = axes[0, idx]
        gdf = df.filter(pl.col("feature_group") == group)

        max_nf = gdf["n_features"].max()
        gdf = gdf.filter(pl.col("n_features") == max_nf)

        mos_df = gdf.filter(pl.col("backend") == "mosaique")
        workers = sorted(mos_df["n_workers"].unique().to_list())
        available_nfiles = sorted(gdf["n_files"].unique().to_list())

        # Pick up to 3 representative file counts
        if len(available_nfiles) >= 3:
            file_counts = [available_nfiles[0], available_nfiles[len(available_nfiles) // 2], available_nfiles[-1]]
        else:
            file_counts = available_nfiles

        bar_width = 0.8 / len(file_counts)
        x = np.arange(len(workers))

        for fi, nf in enumerate(file_counts):
            medians = []
            for w in workers:
                subset = mos_df.filter(
                    (pl.col("n_workers") == w) & (pl.col("n_files") == nf)
                )
                medians.append(subset["wall_s"].median() if len(subset) > 0 else 0)
            offset = (fi - len(file_counts) / 2 + 0.5) * bar_width
            ax.bar(
                x + offset,
                medians,
                bar_width,
                label=f"{nf} file{'s' if nf > 1 else ''}",
                alpha=0.85,
            )

        # MNE baseline (dashed line for max n_files)
        mne_max = gdf.filter(
            (pl.col("backend") == "mne")
            & (pl.col("n_files") == available_nfiles[-1])
        )
        if len(mne_max) > 0:
            mne_median = mne_max["wall_s"].median()
            ax.axhline(mne_median, color=COLOR_MNE, linestyle="--", linewidth=1.5, label="MNE baseline")

        ax.set_xticks(x)
        ax.set_xticklabels([str(w) for w in workers], fontsize=TICK_SIZE)
        ax.set_xlabel("N workers", fontsize=LABEL_SIZE)
        ax.set_ylabel("Wall time (s)", fontsize=LABEL_SIZE)
        ax.set_title(GROUP_LABELS.get(group, group), fontsize=TITLE_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.legend(fontsize=8)

    fig.suptitle("Worker Scaling", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _footnote(axes[0, 0], df)
    fig.savefig(output_dir / "worker_scaling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Feature Scaling
# ---------------------------------------------------------------------------

def plot_feature_scaling(df: pl.DataFrame, output_dir: Path) -> None:
    """Line plot per group: wall time vs n_features at fixed n_files."""
    groups: list[str] = df["feature_group"].unique().drop_nulls().sort().to_list()
    n_groups = len(groups)

    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4), squeeze=False)

    for idx, group in enumerate(groups):
        ax = axes[0, idx]
        gdf = df.filter(pl.col("feature_group") == group)

        # Use max n_files
        max_nfiles = gdf["n_files"].max()
        gdf = gdf.filter(pl.col("n_files") == max_nfiles)

        # MNE line
        mne_stats = (
            gdf.filter(pl.col("backend") == "mne")
            .group_by("n_features")
            .agg(pl.col("wall_s").median().alias("median"))
            .sort("n_features")
        )
        if len(mne_stats) > 0:
            ax.plot(
                mne_stats["n_features"].to_list(),
                mne_stats["median"].to_list(),
                label="MNE",
                color=COLOR_MNE,
                linestyle="--",
                marker="o",
            )

        # Mosaique: use max workers
        mos_df = gdf.filter(pl.col("backend") == "mosaique")
        if len(mos_df) > 0:
            max_w = int(mos_df["n_workers"].max())  # type: ignore[arg-type]
            mos_stats = (
                mos_df.filter(pl.col("n_workers") == max_w)
                .group_by("n_features")
                .agg(pl.col("wall_s").median().alias("median"))
                .sort("n_features")
            )
            ax.plot(
                mos_stats["n_features"].to_list(),
                mos_stats["median"].to_list(),
                label=f"Mosaique {max_w}w",
                color=COLOR_MOS.get(max_w, COLOR_MOS_DEFAULT),
                marker="s",
            )

        ax.set_xlabel("N features", fontsize=LABEL_SIZE)
        ax.set_ylabel("Wall time (s)", fontsize=LABEL_SIZE)
        ax.set_title(f"{GROUP_LABELS.get(group, group)} (n_files={max_nfiles})", fontsize=TITLE_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.legend(fontsize=9)

    fig.suptitle("Feature Scaling", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _footnote(axes[0, 0], df)
    fig.savefig(output_dir / "feature_scaling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Memory Comparison
# ---------------------------------------------------------------------------

def plot_memory_comparison(df: pl.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart: peak RSS by group, MNE vs Mosaique 1w/4w."""
    groups: list[str] = df["feature_group"].unique().drop_nulls().sort().to_list()

    # Filter to max n_files and max n_features per group
    rows: list[dict] = []
    for group in groups:
        gdf = df.filter(pl.col("feature_group") == group)
        max_nf = gdf["n_files"].max()
        max_feat = gdf["n_features"].max()
        gdf = gdf.filter(
            (pl.col("n_files") == max_nf) & (pl.col("n_features") == max_feat)
        )

        # MNE
        mne_rss = gdf.filter(pl.col("backend") == "mne")["peak_rss_mb"].median()
        rows.append({"group": group, "label": "MNE", "rss": mne_rss})

        # Mosaique per worker count
        mos_df = gdf.filter(pl.col("backend") == "mosaique")
        workers = sorted(mos_df["n_workers"].unique().to_list())
        for w in workers:
            rss = mos_df.filter(pl.col("n_workers") == w)["peak_rss_mb"].median()
            rows.append({"group": group, "label": f"Mosaique {w}w", "rss": rss})

    rdf = pl.DataFrame(rows)
    labels: list[str] = rdf["label"].unique().drop_nulls().sort().to_list()

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(groups)), 4.5))

    x = np.arange(len(groups))
    bar_width = 0.8 / len(labels)

    for li, label in enumerate(labels):
        vals = []
        for group in groups:
            subset = rdf.filter((pl.col("group") == group) & (pl.col("label") == label))
            vals.append(subset["rss"][0] if len(subset) > 0 else 0)
        offset = (li - len(labels) / 2 + 0.5) * bar_width
        color = COLOR_MNE if label == "MNE" else COLOR_MOS.get(
            int(label.split()[-1].rstrip("w")), COLOR_MOS_DEFAULT
        )
        ax.bar(x + offset, vals, bar_width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS.get(g, g) for g in groups], fontsize=LABEL_SIZE)
    ax.set_ylabel("Peak RSS (MB)", fontsize=LABEL_SIZE)
    ax.set_title("Memory Comparison (max files, all features)", fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=9)

    fig.tight_layout()
    _footnote(ax, df)
    fig.savefig(output_dir / "memory_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark plots from Parquet results",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input Parquet file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        print("Run benchmark.py first to generate results.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Saving plots to {args.output_dir}/\n")

    print("  [1/5] Speedup heatmap...")
    plot_speedup_heatmap(df, args.output_dir)

    print("  [2/5] Wall time vs N files...")
    plot_wall_vs_nfiles(df, args.output_dir)

    print("  [3/5] Worker scaling...")
    plot_worker_scaling(df, args.output_dir)

    print("  [4/5] Feature scaling...")
    plot_feature_scaling(df, args.output_dir)

    print("  [5/5] Memory comparison...")
    plot_memory_comparison(df, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
