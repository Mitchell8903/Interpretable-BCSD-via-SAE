"""Reproduces jTrans's retrieval baseline using third_party/jTrans/fasteval.py"""

import argparse
import contextlib
import io
import pickle
import random
import re
import statistics
import sys
from pathlib import Path

import torch

from binarycorp import OPTS, key, load_functions, sort_key
from jtrans import embed_functions, load_jtrans, tokenize_functions
from utils import JTRANS_DIR, set_seed, write_result

# The six cross-optimization pairs fasteval.py evaluates.
OPT_PAIRS = [
    ("O0", "O3"),
    ("O0", "Os"),
    ("O1", "Os"),
    ("O1", "O3"),
    ("O2", "Os"),
    ("O2", "O3"),
]


def load_fasteval(poolsize):
    sys.path.insert(0, str(JTRANS_DIR))
    import fasteval

    fasteval.POOLSIZE = poolsize
    return fasteval


def run_pair(fasteval, ebds, type1, type2):
    """MRR and recall@1 for one optimization-level pair."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        recall = fasteval.eval_O(ebds, type1, type2)
    mrr = float(re.search(r"MRR\d+:\s+(\S+)", buf.getvalue()).group(1))
    return {"mrr": mrr, "recall_at_1": float(recall)}


def build_ebds(rows, embeddings):
    """Rows and their embeddings in the pickle format fasteval consumes."""
    groups = {}
    for i, r in enumerate(rows):
        entry = groups.setdefault(
            (r["proj"], r["funcname"]), {"proj": r["proj"], "funcname": r["funcname"]}
        )
        entry[r["opt"]] = embeddings[i].clone().unsqueeze(0)
    return list(groups.values())


def reference_embeddings(ebds):
    """key -> embedding, from the pickle format."""
    return {
        (entry["proj"], entry["funcname"], opt): entry[opt].squeeze(0)
        for entry in ebds
        for opt in OPTS
        if entry.get(opt) is not None
    }


def score(ebds, poolsize, seed):
    """fasteval MRR and recall@1 per pair, and their means."""
    set_seed(seed)
    fasteval = load_fasteval(poolsize)
    pairs = {}
    for type1, type2 in OPT_PAIRS:
        pairs[f"{type1}-{type2}"] = run_pair(fasteval, ebds, type1, type2)
        print(f"{type1}-{type2}: {pairs[f'{type1}-{type2}']}", flush=True)
    mean = {
        m: sum(p[m] for p in pairs.values()) / len(pairs)
        for m in ("mrr", "recall_at_1")
    }
    return pairs, mean


def summarize(runs):
    """Mean and sample sd over seeds of each metrics pair-mean."""
    return {
        m: {
            "mean": statistics.mean(r["mean"][m] for r in runs.values()),
            "sd": statistics.stdev(r["mean"][m] for r in runs.values()),
        }
        for m in ("mrr", "recall_at_1")
    }


def check(rows, ebds, model, tokenizer, n_sample, seed):
    """Cosine between re-embedded sampled functions and the released embeddings."""
    reference = reference_embeddings(ebds)
    # Sorted first: load_functions' order changes with every process.
    shared = sorted((r for r in rows if key(r) in reference), key=sort_key)
    random.seed(seed)
    sample = random.sample(shared, min(n_sample, len(shared)))
    tokens = tokenize_functions(tokenizer, [r["asm"] for r in sample])
    embeddings = embed_functions(model, tokens, tokenizer.pad_token_id)
    expected = torch.stack([reference[key(r)] for r in sample])
    cosine = torch.cosine_similarity(embeddings, expected, dim=1)
    return {
        "n_sample": len(sample),
        "cosine_min": cosine.min().item(),
        "cosine_mean": cosine.mean().item(),
        "max_abs_diff": (embeddings - expected).abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Step 0: jTrans baseline and check")
    parser.add_argument("--dataset_path", default="data/BinaryCorp/small_test")
    parser.add_argument("--model_path", default="models/jTrans-finetune")
    parser.add_argument(
        "--embeddings",
        default="experiments/BinaryCorp-3M/jTrans-finetune.pkl",
        help="written on first run, then reused",
    )
    parser.add_argument("--poolsize", type=int, default=10000)
    parser.add_argument("--n_check", type=int, default=256)
    parser.add_argument("--min_cosine", type=float, default=0.999999)
    # Each seed draws different pools (fasteval shuffles); the check uses the first.
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default="results/baseline.json")
    args = parser.parse_args()

    set_seed(args.seeds[0])
    rows = load_functions(args.dataset_path)
    model, tokenizer = load_jtrans(args.model_path)
    path = Path(args.embeddings)
    if not path.exists():
        tokens = tokenize_functions(tokenizer, [r["asm"] for r in rows])
        embeddings = embed_functions(
            model, tokens, tokenizer.pad_token_id, batch_size=64
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(build_ebds(rows, embeddings)))
    ebds = pickle.loads(path.read_bytes())

    runs = {}
    for seed in args.seeds:
        pairs, mean = score(ebds, args.poolsize, seed)
        runs[str(seed)] = {"pairs": pairs, "mean": mean}
    checked = check(rows, ebds, model, tokenizer, args.n_check, args.seeds[0])
    print(f"check {checked}", flush=True)
    write_result(
        args.out,
        {
            "model_path": args.model_path,
            "embeddings": path.name,
            "poolsize": args.poolsize,
            "seeds": args.seeds,
            "runs": runs,
            "mean": summarize(runs),
            "check": {"min_cosine_threshold": args.min_cosine, **checked},
        },
    )
    assert checked["cosine_min"] >= args.min_cosine, (
        f"jTrans embeddings changed (min cosine {checked['cosine_min']:.8f}): "
        "do not trust any cached activations."
    )


if __name__ == "__main__":
    main()
