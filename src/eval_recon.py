"""Reconstruction fidelity and retrieval recall@1 retained.

Per-token variance explained on held-out token states, and fasteval retrieval
with every asm token's state at the SAE's layer replaced by x_hat and the
blocks above re-run.

    python src/eval_recon.py --layer 6
"""

import argparse
from pathlib import Path

import torch

from activations import read_states, token_mask
from baseline import OPT_PAIRS, build_ebds, load_fasteval, run_pair
from binarycorp import load_index, load_tokens, split_by_project
from jtrans import edited_embeddings, load_jtrans
from sae import load_run
from utils import add_run_args, runs_to_score, score_runs, set_seed, write_result


@torch.no_grad()
def reconstruct(sae, x, scale, batch_size=4096):
    """x_hat for x, on x's own scale."""
    out = []
    for i in range(0, len(x), batch_size):
        _, x_hat = sae(x[i : i + batch_size].cuda() * scale)
        out.append((x_hat / scale).cpu())
    return torch.cat(out)


def retrieval(fasteval, rows, embeddings, seed):
    """Recall@1 and MRR per optimization pair."""
    ebds = build_ebds(rows, embeddings)
    results = {}
    for type1, type2 in OPT_PAIRS:
        set_seed(seed)
        results[f"{type1}-{type2}"] = run_pair(fasteval, ebds, type1, type2)
    return results


def evaluate(run_dir, model, tokenizer, args):
    """Score one SAE and write eval_recon.json."""
    set_seed(args.seed)
    sae, run, scale = load_run(run_dir)
    cfg = run["config"]
    index = load_index(cfg["activations_dir"])
    _, heldout_idx = split_by_project(
        index["functions"], cfg["heldout_frac"], cfg["split_seed"]
    )
    rows = [index["functions"][i] for i in heldout_idx]

    held = token_mask(cfg["activations_dir"], heldout_idx)
    x = read_states(cfg["activations_dir"], cfg["layer"], held).float()
    x_hat = reconstruct(sae, x, scale)
    mse = float(((x - x_hat) ** 2).sum(-1).mean())
    variance = float(((x - x.mean(0)) ** 2).sum(-1).mean())

    tokens = load_tokens(cfg["activations_dir"])[heldout_idx]

    def embed(edit):
        return edited_embeddings(
            model, tokens, cfg["layer"], tokenizer.pad_token_id, edit
        )

    fasteval = load_fasteval(args.poolsize)
    original = retrieval(fasteval, rows, embed(lambda h: h), args.seed)
    rebuilt = embed(lambda h: (sae(h * scale)[1] / scale).to(h.dtype))
    reconstructed = retrieval(fasteval, rows, rebuilt, args.seed)

    pairs = {
        key: {
            "original": original[key],
            "reconstructed": reconstructed[key],
            "retained_recall_at_1": (
                reconstructed[key]["recall_at_1"] / original[key]["recall_at_1"]
            ),
        }
        for key in original
    }
    retained = [p["retained_recall_at_1"] for p in pairs.values()]
    mean_retained = sum(retained) / len(retained)

    out = args.out or str(Path(run_dir) / "eval_recon.json")
    write_result(
        out,
        {
            "run_dir": str(run_dir),
            "config": run["config"],
            "jtrans_commit": run["jtrans_commit"],
            "poolsize": args.poolsize,
            "seed": args.seed,
            "n_heldout": len(x),
            "n_heldout_functions": len(rows),
            "mse": mse,
            "heldout_variance": variance,
            "variance_explained": 1.0 - mse / variance,
            "pairs": pairs,
            "mean_retained_recall_at_1": mean_retained,
        },
    )
    print(
        f"{Path(run_dir).name}: variance explained {1.0 - mse / variance:.4f}  "
        f"retained R@1 {mean_retained:.4f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Metric 1: reconstruction")
    add_run_args(parser, one_config=False)
    parser.add_argument("--poolsize", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, tokenizer = load_jtrans()
    runs = runs_to_score(args, "eval_recon.json")
    score_runs(runs, lambda run_dir: evaluate(run_dir, model, tokenizer, args))


if __name__ == "__main__":
    main()
