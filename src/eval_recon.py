"""Reconstruction fidelity and retrieval recall@1 retained.

Per-token variance explained on held-out token states, and fasteval retrieval
with every asm token's state at the SAE's layer replaced by x_hat and the
blocks above re-run.

The final embedding is read from [CLS], whose unpatched layer state would carry
function information past the SAE. So [CLS] and [SEP] are always replaced by
their mean training-split states, and function information reaches the
embedding only through the asm tokens. Under that ablation, the SAE is scored
against the original asm states and against mean-ablated asm states:
retained_recall_at_1 = reconstructed / original, and
recovered_fraction = (reconstructed - ablated) / (original - ablated).

    python src/eval_recon.py --layer 6
"""

import argparse
import random
from pathlib import Path

import torch

from activations import read_states, token_mask
from baseline import OPT_PAIRS, build_ebds, load_fasteval, run_pair
from binarycorp import load_index, load_tokens, split_by_project
from jtrans import edited_embeddings, load_jtrans, mean_states
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


def recovered(reconstructed, original, ablated):
    return (reconstructed - ablated) / (original - ablated)


_controls = {}


def controls(model, tokenizer, cfg, args):
    """Mean states and the SAE-independent retrieval scores for cfg's
    held-out split, shared by every run on that split."""
    split = (cfg["activations_dir"], cfg["heldout_frac"], cfg["split_seed"], cfg["layer"])
    if split in _controls:
        return _controls[split]
    index = load_index(cfg["activations_dir"])
    train_idx, heldout_idx = split_by_project(
        index["functions"], cfg["heldout_frac"], cfg["split_seed"]
    )
    rows = [index["functions"][i] for i in heldout_idx]
    all_tokens = load_tokens(cfg["activations_dir"])
    tokens = all_tokens[heldout_idx]
    pad = tokenizer.pad_token_id

    sample = random.Random(args.seed).sample(train_idx, args.mean_functions)
    means = mean_states(model, all_tokens[sample], cfg["layer"], pad)
    special = {"cls": means["cls"], "sep": means["sep"]}

    def ablate(h):
        return means["asm"].to(h.dtype).expand_as(h)

    def score(edit, special=None):
        e = edited_embeddings(model, tokens, cfg["layer"], pad, edit, special)
        return retrieval(args.fasteval, rows, e, args.seed)

    _controls[split] = {
        "rows": rows,
        "tokens": tokens,
        "special": special,
        "unedited": score(lambda h: h),
        "original": score(lambda h: h, special),
        "ablated": score(ablate, special),
    }
    return _controls[split]


def score_pairs(original, ablated, reconstructed):
    """Per-pair scores, and the means over pairs."""
    pairs = {
        key: {
            "original": original[key],
            "reconstructed": reconstructed[key],
            "ablated": ablated[key],
            "retained_recall_at_1": (
                reconstructed[key]["recall_at_1"] / original[key]["recall_at_1"]
            ),
            "recovered_fraction": recovered(
                reconstructed[key]["recall_at_1"],
                original[key]["recall_at_1"],
                ablated[key]["recall_at_1"],
            ),
        }
        for key in original
    }
    means = {
        m: sum(p[m] for p in pairs.values()) / len(pairs)
        for m in ("retained_recall_at_1", "recovered_fraction")
    }
    return pairs, means


def evaluate(run_dir, model, tokenizer, args):
    """Score one SAE and write eval_recon.json."""
    set_seed(args.seed)
    sae, run, scale = load_run(run_dir)
    cfg = run["config"]
    ctl = controls(model, tokenizer, cfg, args)

    index = load_index(cfg["activations_dir"])
    _, heldout_idx = split_by_project(
        index["functions"], cfg["heldout_frac"], cfg["split_seed"]
    )
    held = token_mask(cfg["activations_dir"], heldout_idx)
    x = read_states(cfg["activations_dir"], cfg["layer"], held).float()
    x_hat = reconstruct(sae, x, scale)
    mse = float(((x - x_hat) ** 2).sum(-1).mean())
    variance = float(((x - x.mean(0)) ** 2).sum(-1).mean())

    def sae_edit(h):
        return (sae(h * scale)[1] / scale).to(h.dtype)

    e = edited_embeddings(
        model, ctl["tokens"], cfg["layer"], tokenizer.pad_token_id, sae_edit, ctl["special"]
    )
    reconstructed = retrieval(args.fasteval, ctl["rows"], e, args.seed)
    pairs, means = score_pairs(ctl["original"], ctl["ablated"], reconstructed)

    def mean_recall(scores):
        return sum(s["recall_at_1"] for s in scores.values()) / len(scores)

    out = args.out or str(Path(run_dir) / "eval_recon.json")
    write_result(
        out,
        {
            "run_dir": str(run_dir),
            "config": run["config"],
            "jtrans_commit": run["jtrans_commit"],
            "poolsize": args.poolsize,
            "seed": args.seed,
            "mean_functions": args.mean_functions,
            "n_heldout": len(x),
            "n_heldout_functions": len(ctl["rows"]),
            "mse": mse,
            "heldout_variance": variance,
            "variance_explained": 1.0 - mse / variance,
            "pairs": pairs,
            "mean_retained_recall_at_1": means["retained_recall_at_1"],
            "mean_recovered_fraction": means["recovered_fraction"],
            "mean_recall_at_1_unedited": mean_recall(ctl["unedited"]),
            "mean_recall_at_1_original": mean_recall(ctl["original"]),
            "mean_recall_at_1_ablated": mean_recall(ctl["ablated"]),
        },
    )
    print(
        f"{Path(run_dir).name}: variance explained {1.0 - mse / variance:.4f}  "
        f"retained R@1 {means['retained_recall_at_1']:.4f}  "
        f"recovered {means['recovered_fraction']:.4f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Metric 1: reconstruction")
    add_run_args(parser, one_config=False)
    parser.add_argument("--poolsize", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mean_functions",
        type=int,
        default=4096,
        help="training functions averaged for the ablation states",
    )
    args = parser.parse_args()
    args.fasteval = load_fasteval(args.poolsize)

    model, tokenizer = load_jtrans()
    runs = runs_to_score(args, "eval_recon.json")
    score_runs(runs, lambda run_dir: evaluate(run_dir, model, tokenizer, args))


if __name__ == "__main__":
    main()
