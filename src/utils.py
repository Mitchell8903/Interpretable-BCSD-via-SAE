import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from config import SAEConfig

JTRANS_DIR = Path(__file__).resolve().parent.parent / "third_party" / "jTrans"


def set_seed(seed):
    """Seed random/numpy/torch and force deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_result(path, payload):
    """Write a result JSON, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def jtrans_commit():
    """The third_party/jTrans commit."""
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=JTRANS_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def add_run_args(parser, one_config=True):
    """--layer (its sweep) or --run_dir (one SAE), and for one_config the
    --d_sae / --k scored."""
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--layer", type=int, help="score the layer's sweep")
    which.add_argument("--run_dir", help="score one trained SAE")
    if one_config:
        parser.add_argument("--d_sae", type=int, default=SAEConfig.d_sae)
        parser.add_argument("--k", type=int, default=SAEConfig.k)
    parser.add_argument("--out", help="output path (--run_dir only)")


def runs_to_score(args, output_name):
    """The run named by --run_dir, filtered to --d_sae / --k if given."""
    if args.run_dir:
        return [Path(args.run_dir)]
    runs = []
    for run_dir in sorted(SAEConfig(layer=args.layer).sweep_dir().iterdir()):
        if not (run_dir / "run.json").exists() or (run_dir / output_name).exists():
            continue
        cfg = json.loads((run_dir / "run.json").read_text())["config"]
        if "d_sae" in args and (cfg["d_sae"], cfg["k"]) != (args.d_sae, args.k):
            continue
        runs.append(run_dir)
    return runs


def score_runs(runs, evaluate):
    """Call evaluate(run_dir) on each run, reporting failures at the end."""
    print(f"{len(runs)} run(s) to score", flush=True)
    failures = 0
    for i, run_dir in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run_dir.name}", flush=True)
        try:
            evaluate(run_dir)
        except Exception as error:  
            print(f"FAILED {run_dir.name}: {error}", flush=True)
            failures += 1
    if failures:
        raise SystemExit(f"{failures} run(s) failed")
