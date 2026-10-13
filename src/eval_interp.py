"""Autointerp detection accuracy.

    python src/eval_interp.py --layer 6
    python src/eval_interp.py --layer 6 --model gpt-oss:20b
"""

import argparse
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from binarycorp import first_of_each_input, input_keys, key, load_asm, load_index
from binarycorp import load_tokens, split_by_project
from jtrans import load_jtrans, pooled_activations, token_states
from llm import generate, is_openai, load_prompt
from sae import load_run
from utils import add_run_args, runs_to_score, score_runs, set_seed, write_result

NO_PATTERN = "NO CLEAR PATTERN"

DESCRIBE_PROMPT = "describe_feature_v4"
PREDICT_PROMPT = "predict_activation_v1"


def interp_filename(model):
    """eval_interp_<model>.json, so judges never overwrite one another."""
    return f"eval_interp_{model.replace(':', '-')}.json"


@torch.no_grad()
def firing_words(encode, tokens, layer, model, pad_id, feature):
    """Indices of the listing's words where feature fires."""
    ids = tokens.long().cuda()
    states, where = token_states(model, ids, layer, pad_id)
    fires = encode(states.float())[:, feature] > 0
    # Token position p is word p - 1: jTrans tokenizes each space-separated
    # word to exactly one token, after [CLS].
    return [
        (where[(where[:, 0] == row) & fires, 1] - 1).tolist() for row in range(len(ids))
    ]


def highlight(text, words):
    """The listing with each run of firing words wrapped in «»."""
    tokens, fired = text.split(" "), set(words)
    out, i = [], 0
    while i < len(tokens):
        if i not in fired:
            out.append(tokens[i])
            i += 1
            continue
        j = i
        while j + 1 < len(tokens) and j + 1 in fired:
            j += 1
        out.append("«" + " ".join(tokens[i : j + 1]) + "»")
        i = j + 1
    return " ".join(out)


def format_functions(texts, label="Function"):
    """Numbered assembly listings for a prompt."""
    return "\n\n".join(f"### {label} {i + 1}\n{t}" for i, t in enumerate(texts))


def match_negatives(positives, pool, lengths, opts):
    """One non-activating row per positive: same opt level, nearest length."""
    remaining = list(pool)
    chosen = []
    for i in positives:
        same_opt = [j for j in remaining if opts[j] == opts[i]]
        candidates = same_opt or remaining
        best = min(candidates, key=lambda j: abs(lengths[j] - lengths[i]))
        remaining.remove(best)
        chosen.append(best)
    return chosen


def select_contrastive(activations, idx, lengths, opts, n_per_class):
    """Top n_per_class rows plus an equal matched non-activating set, or None."""
    sub = activations[idx]
    nonzero = (sub > 0).nonzero().squeeze(-1).tolist()
    zero_pool = [idx[i] for i in (sub == 0).nonzero().squeeze(-1).tolist()]
    if len(nonzero) < n_per_class or len(zero_pool) < n_per_class:
        return None
    positives = [idx[i] for i in torch.topk(sub, n_per_class).indices.tolist()]
    negatives = match_negatives(positives, zero_pool, lengths, opts)
    return positives, negatives


def describe(pos_texts, neg_texts, args):
    """LLM description of what separates a feature's activations from controls."""
    prompt = load_prompt(DESCRIBE_PROMPT).format(
        n=len(pos_texts),
        positive_examples=format_functions(pos_texts, "A"),
        negative_examples=format_functions(neg_texts, "B"),
    )
    return generate(
        prompt,
        args.model,
        args.cache_dir,
        num_ctx=args.num_ctx,
        reasoning_effort=args.describe_effort,
    ).strip()


def build_candidates(activations, idx, lengths, opts, n_per_class, rng):
    """Balanced, matched (row index, label) pairs in shuffled order."""
    matched = select_contrastive(activations, idx, lengths, opts, n_per_class)
    if matched is None:
        return None
    positives, negatives = matched
    candidates = [(i, 1) for i in positives] + [(i, 0) for i in negatives]
    rng.shuffle(candidates)
    return candidates


def score(description, texts, truth, args):
    """Balanced accuracy of a description at predicting which of texts
    activate (truth, 0/1 each); 0.5 is chance."""
    prompt = load_prompt(PREDICT_PROMPT).format(
        description=description, n=len(texts), candidates=format_functions(texts)
    )
    reply = generate(
        prompt,
        args.model,
        args.cache_dir,
        num_ctx=args.num_ctx,
        reasoning_effort=args.score_effort,
    )

    predicted = [int(d) for d in re.findall(r"[01]", reply)][: len(texts)]
    if len(predicted) != len(texts):
        return None

    n_per_class = sum(truth)
    tpr = sum(p for p, t in zip(predicted, truth) if t == 1) / n_per_class
    tnr = sum(1 - p for p, t in zip(predicted, truth) if t == 0) / n_per_class
    return 0.5 * (tpr + tnr)


def select_features(counts, n_inputs, n_features, min_rate, max_rate, rng):
    """Features firing often enough to describe, but not on nearly everything."""
    rates = counts / n_inputs
    eligible = ((rates >= min_rate) & (rates <= max_rate)).nonzero()
    eligible = eligible.squeeze(-1).tolist()
    return sorted(rng.sample(eligible, min(n_features, len(eligible)))), len(eligible)


def evaluate(run_dir, args):
    """Score one trained SAE and write eval_interp_<model>.json beside it."""
    sae, run, scale = load_run(run_dir)
    out = args.out or str(Path(run_dir) / interp_filename(args.model))
    score_features(sae, scale, run["config"], run["jtrans_commit"], args, out)


def score_features(encoder, scale, cfg, commit, args, out):
    """Autointerp-score any encoder of token states: an SAE or a baseline."""
    # Takes the encoder rather than a run directory so src/interp_baselines.py
    # scores raw dimensions and PCA through exactly this code.
    set_seed(args.seed)
    rng = random.Random(args.seed)
    index = load_index(cfg["activations_dir"])
    n_functions = len(index["functions"])

    # Deduplicate
    keep = first_of_each_input(index["functions"], input_keys(index))
    rows = [index["functions"][i] for i in keep]

    explain_idx, score_idx = split_by_project(rows, args.score_frac, args.seed)
    print(f"{len(rows)} distinct inputs of {n_functions}", flush=True)
    asm = load_asm(index, rows, cfg["activations_dir"], args.max_chars)
    print(f"explain {len(explain_idx)}  score {len(score_idx)}", flush=True)

    def texts(idx):
        return [asm[key(rows[i])] for i in idx]

    model, tokenizer = load_jtrans()
    pad_id = tokenizer.pad_token_id
    tokens = load_tokens(cfg["activations_dir"])[keep]
    gpu = encoder.to("cuda")

    def encode(h):
        return gpu.encode(h * scale)

    pooled = pooled_activations(encode, tokens, cfg["layer"], model, pad_id)

    counts = sum((chunk > 0).sum(0) for chunk in pooled.split(4096))
    features, n_eligible = select_features(
        counts, len(rows), args.n_features, args.min_rate, args.max_rate, rng
    )
    print(f"{n_eligible} eligible features, scoring {len(features)}", flush=True)
    acts = pooled[:, features].float()  # (n_functions, len(features))

    lengths = [len(t) for t in texts(range(len(rows)))]
    opts = [r["opt"] for r in rows]

    entries = [
        {
            "feature": feature,
            "exemplars": select_contrastive(
                acts[:, column], explain_idx, lengths, opts, args.n_exemplars
            ),
            "candidates": build_candidates(
                acts[:, column], score_idx, lengths, opts, args.n_per_class, rng
            ),
            "description": None,
        }
        for column, feature in enumerate(features)
    ]

    # Skip describing a feature that could never be scored anyway
    describable = [
        e for e in entries if e["exemplars"] is not None and e["candidates"] is not None
    ]
    print(f"describing {len(describable)}/{len(entries)} features", flush=True)

    for e in describable:
        positives, negatives = e["exemplars"]
        words = firing_words(
            encode, tokens[positives], cfg["layer"], model, pad_id, e["feature"]
        )
        highlighted = [highlight(t, w) for t, w in zip(texts(positives), words)]
        e["texts"] = (highlighted, texts(negatives))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        descriptions = pool.map(lambda e: describe(*e["texts"], args), describable)
        for e, description in zip(describable, descriptions):
            e["description"] = description

    patterned = [e for e in describable if NO_PATTERN not in e["description"].upper()]
    print(f"scoring {len(patterned)}/{len(entries)} features", flush=True)

    def score_entry(e):
        """A feature's own description and a mismatched one on the same candidates."""
        mismatched = patterned[(patterned.index(e) + 1) % len(patterned)]
        candidates = texts([i for i, _ in e["candidates"]])
        truth = [label for _, label in e["candidates"]]
        return (
            score(e["description"], candidates, truth, args),
            score(mismatched["description"], candidates, truth, args),
        )

    scores = {}
    if len(patterned) > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for e, pair in zip(patterned, pool.map(score_entry, patterned)):
                scores[e["feature"]] = pair

    results = []
    for e in entries:
        detection, control = scores.get(e["feature"], (None, None))
        results.append(
            {
                "feature": e["feature"],
                "firing_rate": float(counts[e["feature"]] / len(rows)),
                "description": e["description"],
                "no_pattern": e["description"] is not None
                and NO_PATTERN in e["description"].upper(),
                "too_few_activations": e["exemplars"] is None
                or e["candidates"] is None,
                "detection_acc": detection,
                "control_acc": control,
            }
        )

    paired = [
        r
        for r in results
        if r["detection_acc"] is not None and r["control_acc"] is not None
    ]

    def mean(key):
        return sum(r[key] for r in paired) / len(paired) if paired else None

    detection, control = mean("detection_acc"), mean("control_acc")

    write_result(
        out,
        {
            "config": cfg,
            "jtrans_commit": commit,
            "llm": args.model,
            "reasoning_effort": (
                {"describe": args.describe_effort, "score": args.score_effort}
                if is_openai(args.model)
                else None
            ),
            "settings": {
                name: getattr(args, name)
                for name in [
                    "n_features",
                    "n_exemplars",
                    "n_per_class",
                    "max_chars",
                    "score_frac",
                    "min_rate",
                    "max_rate",
                ]
            },
            "prompts": [DESCRIBE_PROMPT, PREDICT_PROMPT],
            "seed": args.seed,
            "n_functions": n_functions,
            "n_distinct_inputs": len(rows),
            "n_explain": len(explain_idx),
            "n_score": len(score_idx),
            "n_eligible_features": n_eligible,
            "n_features": len(features),
            "n_no_pattern": sum(r["no_pattern"] for r in results),
            "n_too_few_activations": sum(r["too_few_activations"] for r in results),
            "n_scored": len(paired),
            "detection_acc": detection,
            "control_acc": control,
            "features": results,
        },
    )

    summary = (
        f"detection {detection:.4f}  control {control:.4f}"
        if paired
        else "nothing scorable"
    )
    print(
        f"{summary}  ({len(paired)} scored, "
        f"{sum(r['no_pattern'] for r in results)} no-pattern, "
        f"{sum(r['too_few_activations'] for r in results)} too-rare)",
        flush=True,
    )
    print(f"wrote {out}", flush=True)


def add_interp_args(parser):
    """The judge and sampling flags, shared with interp_baselines.py."""
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--n_features", type=int, default=150)
    parser.add_argument("--n_exemplars", type=int, default=20)
    parser.add_argument("--n_per_class", type=int, default=10)
    parser.add_argument("--max_chars", type=int, default=4000)
    parser.add_argument("--num_ctx", type=int, default=65536)
    parser.add_argument("--describe_effort", default="medium")
    parser.add_argument("--score_effort", default="low")
    parser.add_argument("--score_frac", type=float, default=0.5)
    parser.add_argument("--min_rate", type=float, default=0.001)
    parser.add_argument("--max_rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache_dir", default="data/llm_cache")


def main():
    parser = argparse.ArgumentParser(description="Metric 2: autointerp")
    add_run_args(parser)
    add_interp_args(parser)
    args = parser.parse_args()

    runs = runs_to_score(args, interp_filename(args.model))
    score_runs(runs, lambda run_dir: evaluate(run_dir, args))


if __name__ == "__main__":
    main()
