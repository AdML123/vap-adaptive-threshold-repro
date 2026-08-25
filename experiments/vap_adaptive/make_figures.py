"""Create publication figures from immutable experiment artifacts.

The script intentionally reads only saved CSV/JSON artifacts; it never
recomputes labels or metrics from raw audio.  Matplotlib/Agg is used for all
rendering so the figure bundle is reproducible on headless workers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)


def save_pub(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_figure_1(repo: Path, output: Path) -> dict:
    rows = [
        r
        for r in read_csv(repo / "experiments/bc_erase/results/summary.csv")
        if r["variant"] == "median_laughter_voiced" and r["split"] == "test"
    ]
    rows.sort(key=lambda r: int(r["bin_k"]))
    bins = np.arange(1, 5)
    widths = np.array([200, 400, 600, 800])
    d = np.array([float(r["D_k"]) for r in rows])
    dom = np.array([float(r["D_k_dom"]) for r in rows])
    coex = np.array([float(r["D_k_coex"]) for r in rows])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.9, 2.55), gridspec_kw={"wspace": 0.34})
    ax0.plot(widths, d, marker="o", color="#1f4e79", lw=1.7, ms=4.5, label="all BC bins")
    ax0.plot(widths, dom, marker="o", color="#666666", lw=1.4, ms=4, label="bc-dominant")
    ax0.plot(widths, coex, marker="o", color="#c44e52", lw=1.7, ms=4.5, label="bc-coexisting")
    ax0.set(xlabel="Bin width (ms)", ylabel="Label-erasure rate, $D_k$", ylim=(0, 0.92), xticks=widths)
    ax0.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax0.legend(loc="upper left", fontsize=6.4, handlelength=1.8)
    ax0.text(-0.14, 1.04, "a", transform=ax0.transAxes, fontweight="bold", fontsize=9)

    x = np.arange(4)
    barw = 0.34
    ax1.bar(x - barw / 2, dom, width=barw, color="#9a9a9a", label="bc-dominant")
    ax1.bar(x + barw / 2, coex, width=barw, color="#c44e52", label="bc-coexisting")
    for idx, (a, b) in enumerate(zip(dom, coex)):
        ax1.text(idx + barw / 2, b + 0.018, f"{(b-a)*100:.0f} pp", ha="center", va="bottom", fontsize=6)
    ax1.set(xlabel="Projection bin", ylabel="Label-erasure rate", ylim=(0, 0.98), xticks=x, xticklabels=["1\n200", "2\n400", "3\n600", "4\n800"])
    ax1.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax1.legend(loc="upper left", fontsize=6.4, handlelength=1.8)
    ax1.text(-0.14, 1.04, "b", transform=ax1.transAxes, fontweight="bold", fontsize=9)
    fig.suptitle("Label erasure increases with bin width and speaker coexistence", y=1.02, fontsize=8.5)
    save_pub(fig, output / "fig1_erasure")
    return {"source": "experiments/bc_erase/results/summary.csv", "variant": "median_laughter_voiced", "split": "test", "rows": rows}


def read_metrics(path: Path) -> list[dict[str, float]]:
    rows = []
    for r in read_csv(path):
        if not r.get("epoch") or not r.get("val_loss"):
            continue
        rows.append({k: float(v) for k, v in r.items() if v not in (None, "")})
    return rows


def make_figure_2(repo: Path, output: Path) -> dict:
    paths = {
        "standard": repo / "experiments/vap_adaptive/results/standard/1/logs/metrics.csv",
        "adaptive forward": repo / "experiments/vap_adaptive/results/adaptive_forward/1/logs/metrics.csv",
    }
    series = {label: read_metrics(path) for label, path in paths.items()}
    colors = {"standard": "#666666", "adaptive forward": "#1f4e79"}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.9, 2.55), gridspec_kw={"wspace": 0.34})
    for label, rows in series.items():
        epochs = np.array([r["epoch"] + 1 for r in rows])
        loss = np.array([r["val_loss"] for r in rows])
        entropy = np.array([r["val_target_entropy"] for r in rows])
        ax0.plot(epochs, loss, color=colors[label], lw=1.5, label=label)
        ax1.plot(epochs, entropy, color=colors[label], lw=1.5, label=label)
    ax0.set(xlabel="Epoch", ylabel="Validation cross-entropy")
    ax1.set(xlabel="Epoch", ylabel="Target-state entropy (bits)")
    for ax in (ax0, ax1):
        ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
        ax.legend(loc="best", fontsize=6.4, handlelength=1.8)
    ax0.text(-0.14, 1.04, "a", transform=ax0.transAxes, fontweight="bold", fontsize=9)
    ax1.text(-0.14, 1.04, "b", transform=ax1.transAxes, fontweight="bold", fontsize=9)
    fig.suptitle("Adaptive labels preserve training stability", y=1.02, fontsize=8.5)
    save_pub(fig, output / "fig2_stability")
    return {
        "sources": {
            label: path.relative_to(repo).as_posix()
            for label, path in paths.items()
        },
        "series": series,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = (args.output or repo / "experiments/vap_adaptive/results/aggregate_final/figures").resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "backend": "matplotlib_agg",
        "archetype": "quantitative_grid",
        "figure_1": make_figure_1(repo, output),
        "figure_2": make_figure_2(repo, output),
        "source_data_policy": "saved experiment artifacts only",
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "figures": ["fig1_erasure", "fig2_stability"]}, indent=2))


if __name__ == "__main__":
    main()
