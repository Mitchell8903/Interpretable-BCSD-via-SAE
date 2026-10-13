"""Train an SAE on one layer's cached asm-token states.

    python src/train_sae.py --layer 6 --d_sae 4096 --k 32 --seed 0
    python src/train_sae.py --layer 6 --sweep
"""

import argparse
import dataclasses

import torch

from activations import read_states, token_mask
from binarycorp import load_index, training_inputs
from config import SWEEP_D_SAE, SWEEP_K, SWEEP_SEEDS, SAEConfig
from sae import SAE, auxk_loss, sae_loss, topk
from utils import jtrans_commit, set_seed, write_result


def load_activations(cfg):
    """token states of the training and held-out functions."""
    train_idx, _ = training_inputs(
        load_index(cfg.activations_dir), cfg.heldout_frac, cfg.split_seed
    )
    is_train = token_mask(cfg.activations_dir, train_idx)
    return (
        read_states(cfg.activations_dir, cfg.layer, is_train),
        read_states(cfg.activations_dir, cfg.layer, ~is_train),
    )


def normalization_scale(x, batch_size=65536):
    """Scale making E||x||_2 = sqrt(d_model)"""
    norms = torch.cat(
        [
            x[i : i + batch_size].float().norm(dim=1)
            for i in range(0, len(x), batch_size)
        ]
    )
    return (x.shape[1] ** 0.5) / norms.mean()


@torch.no_grad()
def evaluate(sae, x, scale, batch_size):
    """Mean recon loss, mean L0 and dead-feature fraction over x."""
    total_recon = 0.0
    total_l0 = 0.0
    fired = torch.zeros(sae.b_enc.shape[0], dtype=torch.bool, device=x.device)
    for i in range(0, len(x), batch_size):
        batch = x[i : i + batch_size].float() * scale
        z, x_hat = sae(batch)
        total_recon += sae_loss(batch, x_hat).item() * len(batch)
        total_l0 += (z > 0).sum(-1).float().sum().item()
        fired |= (z > 0).any(0)
    return {
        "recon": total_recon / len(x),
        "l0": total_l0 / len(x),
        "dead_frac": 1.0 - fired.float().mean().item(),
    }


def train(cfg, device="cuda"):
    """Train one SAE (sae, epoch log, scale, d_model, n_train)."""
    set_seed(cfg.seed)
    x_train, x_heldout = load_activations(cfg)

    scale = normalization_scale(x_train)
    x_train, x_heldout = x_train.to(device), x_heldout.to(device)
    print(f"train {tuple(x_train.shape)}  heldout {tuple(x_heldout.shape)}", flush=True)

    d_model = x_train.shape[1]
    sae = SAE(d_model, cfg.d_sae, cfg.k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    tokens_since_fired = torch.zeros(cfg.d_sae, device=device)

    log = []
    for epoch in range(cfg.epochs):
        order = torch.randperm(len(x_train), device=device)
        for i in range(0, len(order), cfg.batch_size):
            batch = x_train[order[i : i + cfg.batch_size]].float() * scale
            pre = sae.pre_activations(batch)  # (batch, d_sae)
            z = topk(pre, cfg.k)
            x_hat = sae.decode(z)
            dead = tokens_since_fired >= cfg.dead_tokens
            loss = sae_loss(batch, x_hat) + cfg.aux_coef * auxk_loss(
                sae, batch, x_hat, pre, dead, cfg.k_aux
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

            tokens_since_fired += len(batch)
            tokens_since_fired[(z > 0).any(0)] = 0

        metrics = evaluate(sae, x_heldout, scale, cfg.batch_size)
        metrics["epoch"] = epoch
        log.append(metrics)
        print(
            f"epoch {epoch:3d}  recon {metrics['recon']:.4f}  "
            f"L0 {metrics['l0']:.1f}  dead {metrics['dead_frac']:.3f}",
            flush=True,
        )

    return sae, log, float(scale), d_model, len(x_train)


def train_and_save(cfg):
    """Train one SAE and write sae.pt and run.json."""
    sae, log, scale, d_model, n_train = train(cfg)

    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sae.state_dict(), "scale": scale}, run_dir / "sae.pt")
    write_result(
        run_dir / "run.json",
        {
            "config": dataclasses.asdict(cfg),
            "d_model": d_model,
            "n_train": n_train,
            "activation_scale": scale,
            "jtrans_commit": jtrans_commit(),
            "log": log,
            "final": log[-1],
        },
    )
    print(f"wrote {run_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Step 2: train SAEs")
    for field in dataclasses.fields(SAEConfig):
        parser.add_argument(
            f"--{field.name}",
            type=type(field.default),
            help=f"default {field.default!r}",
        )
    parser.add_argument(
        "--sweep", action="store_true", help="train every seed x d_sae x k"
    )
    args = parser.parse_args()

    names = {f.name for f in dataclasses.fields(SAEConfig)}
    overrides = {k: v for k, v in vars(args).items() if k in names and v is not None}

    if not args.sweep:
        train_and_save(SAEConfig(**overrides))
        return

    grid = [
        {"seed": seed, "d_sae": d_sae, "k": k}
        for seed in SWEEP_SEEDS
        for d_sae in SWEEP_D_SAE
        for k in SWEEP_K
    ]
    for i, point in enumerate(grid, 1):
        cfg = SAEConfig(**{**overrides, **point})
        if (cfg.run_dir() / "run.json").exists():
            print(f"[{i}/{len(grid)}] have {cfg.name()}, skipping", flush=True)
            continue
        print(f"[{i}/{len(grid)}] {cfg.name()}", flush=True)
        train_and_save(cfg)


if __name__ == "__main__":
    main()
