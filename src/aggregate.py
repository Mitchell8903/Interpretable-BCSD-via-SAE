"""Aggregate one results from one layer's sweep into a summary JSON.

Reads each run's run.json and whichever metric files exist, groups runs by
(d_sae, k), and writes results/layer<L>/summary.json.

    python src/aggregate.py --layer 6
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from config import SAEConfig
from eval_interp import interp_filename
from utils import jtrans_commit, write_result


TRAIN_METRICS = ["l0", "dead_frac", "recon"]
RECON_METRICS = ["variance_explained", "mean_retained_recall_at_1"]
INTERP_METRICS = ["detection_acc", "control_acc", "n_scored"]
OPTIONAL_METRICS = RECON_METRICS + INTERP_METRICS + ["detection_acc_unscored_at_chance"]


def summarize(values):
    """Mean and sample sd; sd is None for one seed."""
    sd = statistics.stdev(values) if len(values) > 1 else None
    return {"mean": statistics.mean(values), "sd": sd}


def unscored_at_chance(interp):
    """Detection accuracy over every sampled feature, unscored ones at 0.5."""
    n_scored, n_features = interp["n_scored"], interp["n_features"]
    total = interp["detection_acc"] * n_scored + 0.5 * (n_features - n_scored)
    return total / n_features


def interp_columns(interp):
    columns = {m: interp[m] for m in INTERP_METRICS}
    columns["detection_acc_unscored_at_chance"] = unscored_at_chance(interp)
    return columns


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def run_columns(run_dir, judge):
    """One run's metrics, training plus whichever metric files exist."""
    run = read_json(run_dir / "run.json")
    columns = {m: run["final"][m] for m in TRAIN_METRICS}
    recon = read_json(run_dir / "eval_recon.json")
    if recon is not None:
        columns.update({m: recon[m] for m in RECON_METRICS})
    interp = read_json(run_dir / interp_filename(judge))
    if interp is not None:
        columns.update(interp_columns(interp))
    return run["config"], columns


def baselines(baselines_dir, judge):
    """interpretability baselines, with a standard error over their features."""
    out = {}
    for path in sorted(Path(baselines_dir).glob(f"*/{interp_filename(judge)}")):
        interp = read_json(path)
        accs = [
            f["detection_acc"]
            for f in interp["features"]
            if f["detection_acc"] is not None and f["control_acc"] is not None
        ]
        row = interp_columns(interp)
        row["detection_acc_se"] = statistics.stdev(accs) / len(accs) ** 0.5
        out[interp["config"]["baseline"]] = row
    return out


def aggregate(sweep_dir, judge):
    """Rows aggregated over seeds, one per (d_sae, k)."""
    by_setting = defaultdict(list)
    for run_dir in sorted(p.parent for p in Path(sweep_dir).glob("*/run.json")):
        cfg, columns = run_columns(run_dir, judge)
        by_setting[(cfg["d_sae"], cfg["k"])].append(columns)
    rows = []
    for (d_sae, k), runs in sorted(by_setting.items()):
        row = {"d_sae": d_sae, "k": k, "n_seeds": len(runs)}
        # A metric is reported only where every seed has it.
        present = [m for m in OPTIONAL_METRICS if all(m in r for r in runs)]
        for m in TRAIN_METRICS + present:
            row[m] = summarize([r[m] for r in runs])
        rows.append(row)
    return rows


def print_row(r):
    line = f"d_sae {r['d_sae']:<6} k {r['k']:<3} dead {r['dead_frac']['mean']:.3f}"
    if "variance_explained" in r:
        line += (
            f"  var_exp {r['variance_explained']['mean']:.3f}"
            f"  retained R@1 {r['mean_retained_recall_at_1']['mean']:.3f}"
        )
    if "detection_acc" in r:
        line += (
            f"  detection {r['detection_acc']['mean']:.3f}"
            f" (control {r['control_acc']['mean']:.3f})"
        )
    print(line, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Aggregate one layer's sweep")
    parser.add_argument("--layer", type=int, default=SAEConfig.layer)
    parser.add_argument(
        "--judge", default="gpt-6-luna", help="whose metric-2 files to read"
    )
    args = parser.parse_args()

    layer_dir = SAEConfig(layer=args.layer).sweep_dir().parent
    rows = aggregate(layer_dir / "sweep", args.judge)
    base = baselines(layer_dir / "baselines", args.judge)
    write_result(
        layer_dir / "summary.json",
        {
            "layer": args.layer,
            "judge": args.judge,
            "jtrans_commit": jtrans_commit(),
            "rows": rows,
            "baselines": base,
        },
    )
    for r in rows:
        print_row(r)
    for kind, b in base.items():
        print(
            f"baseline {kind:<5} detection {b['detection_acc']:.3f}"
            f" +/- {b['detection_acc_se']:.3f} (control {b['control_acc']:.3f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
