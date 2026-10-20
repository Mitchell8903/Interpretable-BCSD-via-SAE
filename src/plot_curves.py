"""Per-epoch curves, mean +/- sd over seeds, one panel per (d_sae, k).

Takes run.json files, or directories holding them (a run directory or a whole
sweep). The curve is the per-epoch held-out metric each run logs. Each panel's
y-axis is zoomed past the first 10% of epochs, where seed spread is otherwise
too small to see; --full_y shares one y-axis covering every epoch.

    python src/plot_curves.py results/layer6/sweep
    python src/plot_curves.py results/layer6/sweep --metric dead_frac --out dead.pdf
    python src/plot_curves.py results/layer6/sweep/dsae16384_k*_seed*/run.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "recon": "Held-out reconstruction MSE",
    "dead_frac": "Dead-feature fraction",
    "l0": "L0",
}
LINE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def run_files(paths):
    """Every run.json named by paths, expanding directories."""
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += [p / "run.json"] if (p / "run.json").exists() else sorted(p.glob("*/run.json"))
        else:
            files.append(p)
    return files


def group_curves(files, metric):
    """{(d_sae, k): (seeds x epochs array, config of the first run)}."""
    curves, configs = defaultdict(list), {}
    for f in files:
        run = json.loads(Path(f).read_text())
        cfg = run["config"]
        curves[(cfg["d_sae"], cfg["k"])].append([e[metric] for e in run["log"]])
        configs.setdefault((cfg["d_sae"], cfg["k"]), cfg)
    out = {}
    for setting, runs in curves.items():
        # Runs cut short would otherwise break the stack.
        n = min(map(len, runs))
        out[setting] = (np.array([r[:n] for r in runs]), configs[setting])
    return out


def plot(grouped, metric, out, full_y=False):
    d_saes = sorted({d for d, _ in grouped})
    ks = sorted({k for _, k in grouped})
    fig, axes = plt.subplots(
        len(d_saes), len(ks), figsize=(max(3.2 * len(ks), 6.0), 2.5 * len(d_saes)),
        sharex=True, sharey=full_y, squeeze=False,
    )
    for i, d_sae in enumerate(d_saes):
        for j, k in enumerate(ks):
            ax = axes[i][j]
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.grid(color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            if (d_sae, k) not in grouped:
                ax.set_title(f"d_sae {d_sae}, k {k}: no runs", fontsize=9, color=MUTED)
                continue
            values, cfg = grouped[(d_sae, k)]
            epochs = np.arange(1, values.shape[1] + 1)
            mean = values.mean(0)
            sd = values.std(0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
            if len(values) > 1:
                ax.fill_between(epochs, mean - sd, mean + sd, color=LINE, alpha=0.25, linewidth=0)
            if not full_y:
                tail = slice(len(mean) // 10, None)
                lo, hi = (mean - sd)[tail].min(), (mean + sd)[tail].max()
                pad = 0.08 * (hi - lo) or 0.05 * abs(hi) or 1.0
                ax.set_ylim(lo - pad, hi + pad)
            ax.plot(epochs, mean, color=LINE, linewidth=1.6)
            if cfg.get("lr_decay_frac", 0) > 0:
                ax.axvline(cfg["epochs"] * (1 - cfg["lr_decay_frac"]), color=MUTED,
                           linewidth=0.8, linestyle="--")
            ax.set_title(f"d_sae {d_sae}, k {k}  (n={len(values)})", fontsize=9, color=INK)
            ax.annotate(f"{mean[-1]:.3g}", (epochs[-1], mean[-1]), xytext=(-2, 6),
                        textcoords="offset points", ha="right", fontsize=8, color=INK)
        axes[i][0].set_ylabel(LABELS.get(metric, metric), fontsize=8, color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("Epoch", fontsize=8, color=INK)
    notes = []
    if any(len(values) > 1 for values, _ in grouped.values()):
        notes.append("Line: mean over seeds. Band: ±1 sd.")
    if any(cfg.get("lr_decay_frac", 0) > 0 for _, cfg in grouped.values()):
        notes.append("Dashed: lr decay starts.")
    if not full_y:
        notes.append("y-axes zoomed past the first 10% of epochs.")
    fig.text(0.5, 0.005, " ".join(notes), ha="center", fontsize=8, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Plot per-epoch curves")
    parser.add_argument("paths", nargs="+", help="run.json files or directories")
    parser.add_argument("--metric", default="recon", choices=sorted(LABELS))
    parser.add_argument("--out", help="default curves_<metric>.png")
    parser.add_argument(
        "--full_y", action="store_true", help="one shared y-axis over every epoch"
    )
    args = parser.parse_args()
    files = run_files(args.paths)
    if not files:
        raise SystemExit("no run.json found")
    plot(
        group_curves(files, args.metric),
        args.metric,
        args.out or f"curves_{args.metric}.png",
        args.full_y,
    )


if __name__ == "__main__":
    main()
