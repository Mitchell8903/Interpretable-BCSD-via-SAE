"""Interpretability baselines (raw residual dimensions and PCA components).

    python src/interp_baselines.py --layer 6
"""

import argparse

import numpy as np
import torch
from sklearn.decomposition import PCA

from activations import read_states, token_mask
from binarycorp import load_index, training_inputs
from config import SAEConfig
from eval_interp import add_interp_args, interp_filename, score_features
from utils import jtrans_commit


class SignedDirections:
    """Each direction as two features: above its high threshold, below its low."""

    def __init__(self, directions, high, low):
        self.directions = directions  # (n_directions, d_model)
        self.high, self.low = high, low  # (n_directions,) each

    def encode(self, x):
        """Split-sign projections, shape (batch, 2 * n_directions)."""
        proj = x @ self.directions.T
        return torch.cat(
            [torch.relu(proj - self.high), torch.relu(self.low - proj)], dim=-1
        )

    def to(self, device):
        return SignedDirections(
            self.directions.to(device), self.high.to(device), self.low.to(device)
        )


def fit_baseline(kind, x_train, fire_rate, n_features):
    """Raw basis or PCA components, each side firing on fire_rate of x_train."""
    if kind == "raw":
        directions = torch.eye(x_train.shape[1])
    else:
        n_components = n_features // 2 if kind == "pca_top" else None
        pca = PCA(n_components=n_components, svd_solver="full").fit(x_train.numpy())
        directions = torch.from_numpy(pca.components_).float()
    proj = x_train @ directions.T  # (n, n_directions)
    n = len(proj)
    high = proj.kthvalue(n - int(n * fire_rate), dim=0).values
    low = proj.kthvalue(max(int(n * fire_rate), 1), dim=0).values
    return SignedDirections(directions, high, low)


def fitting_states(cfg, n_tokens, seed):
    """A sample of the token states the SAE trains on, (n_tokens, d_model)."""
    # All 4.4M would not fit PCA in memory.
    train_idx, _ = training_inputs(
        load_index(cfg.activations_dir), cfg.heldout_frac, cfg.split_seed
    )
    is_train = token_mask(cfg.activations_dir, train_idx)
    rows = is_train.nonzero()[0]
    sample = np.random.default_rng(seed).choice(rows, n_tokens, replace=False)
    keep = np.zeros(len(is_train), dtype=bool)
    keep[sample] = True
    return read_states(cfg.activations_dir, cfg.layer, keep).float()


def main():
    parser = argparse.ArgumentParser(description="Metric 2 baselines: raw and PCA")
    parser.add_argument("--layer", type=int, default=SAEConfig.layer)
    parser.add_argument("--kinds", nargs="+", default=["raw", "pca", "pca_top"])
    parser.add_argument(
        "--fit_tokens", type=int, default=500_000, help="training states to fit on"
    )
    # The SAE config whose sparsity the baselines match (activation rate k / d_sae).
    parser.add_argument("--d_sae", type=int, default=SAEConfig.d_sae)
    parser.add_argument("--k", type=int, default=SAEConfig.k)
    add_interp_args(parser)
    args = parser.parse_args()

    cfg = SAEConfig(layer=args.layer)
    fire_rate = args.k / args.d_sae
    print(f"matching d_sae {args.d_sae}, k {args.k}: fire rate {fire_rate:.4f}")
    x_fit = fitting_states(cfg, args.fit_tokens, args.seed)

    for kind in args.kinds:
        out = cfg.sweep_dir().parent / "baselines" / kind / interp_filename(args.model)
        if out.exists():
            print(f"have {out}, skipping", flush=True)
            continue
        print(f"baseline {kind}", flush=True)
        config = {
            "baseline": kind,
            "layer": cfg.layer,
            "fire_rate": fire_rate,
            "activations_dir": cfg.activations_dir,
            "heldout_frac": cfg.heldout_frac,
            "split_seed": cfg.split_seed,
        }
        encoder = fit_baseline(kind, x_fit, fire_rate, args.n_features)
        score_features(encoder, 1.0, config, jtrans_commit(), args, out)


if __name__ == "__main__":
    main()
