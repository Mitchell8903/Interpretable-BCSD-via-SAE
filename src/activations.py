"""Cache jTrans's residual stream at sampled asm-token positions.

Each asm token's state at a layer is one SAE training example. 
The first run also writes the function indexand token ids that 
every later step reads.

    python src/activations.py --layers 6 --positions 64
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from binarycorp import INDEX_FILE, TOKENS_FILE, load_functions, load_index
from binarycorp import load_tokens, sort_key, training_inputs
from config import SWEEP_LAYERS, SAEConfig
from jtrans import load_jtrans, tokenize_functions
from utils import set_seed, write_result


def build_index(activations_dir, dataset_path, tokenizer):
    """Write the function index and token ids if missing; return the index."""
    directory = Path(activations_dir)
    if (directory / INDEX_FILE).exists() and (directory / TOKENS_FILE).exists():
        return load_index(activations_dir)
    rows = sorted(load_functions(dataset_path), key=sort_key)
    write_result(
        directory / INDEX_FILE,
        {
            "description": (
                "Canonical row order of BinaryCorp small_test functions; "
                "every cache indexes into it."
            ),
            "dataset_path": dataset_path,
            "functions": [{k: r[k] for k in ("proj", "funcname", "opt")} for r in rows],
        },
    )
    tokens = tokenize_functions(tokenizer, [r["asm"] for r in rows])
    torch.save(tokens, directory / TOKENS_FILE)
    return load_index(activations_dir)


def sample_index(tokens, functions, per_function, pad_id, generator):
    """(function row, position) of up to per_function asm tokens each, (n, 2)."""
    index = []
    for i in functions:
        n_real = int((tokens[i] != pad_id).sum())
        positions = torch.arange(1, n_real - 1)
        if len(positions) > per_function:
            keep = torch.randperm(len(positions), generator=generator)[:per_function]
            positions = positions[keep.sort().values]
        index.extend((i, int(p)) for p in positions)
    return torch.tensor(index, dtype=torch.int32)  # (n_tokens, 2)


@torch.no_grad()
def extract_states(model, tokens, index, layers, out, pad_id, batch_size=64):
    """Write each sampled token's state at each layer into out[layer]."""
    functions = index[:, 0].unique_consecutive()
    starts = torch.searchsorted(index[:, 0].contiguous(), functions)
    ends = torch.cat([starts[1:], torch.tensor([len(index)])])
    for b in tqdm(range(0, len(functions), batch_size)):
        batch = functions[b : b + batch_size]
        ids = tokens[batch].long().cuda()
        hidden = model(
            input_ids=ids,
            attention_mask=(ids != pad_id).long(),
            output_hidden_states=True,
        ).hidden_states
        lo, hi = int(starts[b]), int(ends[b + len(batch) - 1])
        # Row of each index entry within this batch of functions.
        local = torch.repeat_interleave(
            torch.arange(len(batch)),
            ends[b : b + len(batch)] - starts[b : b + len(batch)],
        )
        positions = index[lo:hi, 1].long()
        for layer in layers:
            states = hidden[layer][local.cuda(), positions.cuda()]  # (n, d_model)
            out[layer][lo:hi] = states.half().cpu().numpy()


def states_path(activations_dir, layer):
    return Path(activations_dir) / f"layer{layer}_tokens.npy"


def token_mask(activations_dir, rows):
    """Which cached tokens belong to function rows."""
    index = torch.load(Path(activations_dir) / "token_index.pt")
    return torch.isin(index[:, 0], torch.tensor(rows, dtype=torch.int32)).numpy()


def read_states(activations_dir, layer, keep, chunk=1_000_000):
    """The cached token states where keep is True, (n, d_model) float16."""
    x = np.load(states_path(activations_dir, layer), mmap_mode="r")
    out = torch.empty((int(keep.sum()), x.shape[1]), dtype=torch.float16)
    filled = 0
    for i in range(0, len(x), chunk):
        rows = np.array(x[i : i + chunk])[keep[i : i + chunk]]
        out[filled : filled + len(rows)] = torch.from_numpy(rows)
        filled += len(rows)
    return out


def main():
    parser = argparse.ArgumentParser(description="Cache jTrans token activations")
    parser.add_argument("--layers", type=int, nargs="+", default=list(SWEEP_LAYERS))
    parser.add_argument(
        "--positions", type=int, default=64, help="per training function"
    )
    parser.add_argument(
        "--heldout_positions",
        type=int,
        default=8,
        help="per held-out function; only feeds the per-epoch loss",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--activations_dir", default=SAEConfig.activations_dir)
    parser.add_argument("--dataset_path", default="data/BinaryCorp/small_test")
    args = parser.parse_args()

    set_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    cfg = SAEConfig()
    model, tokenizer = load_jtrans()
    function_index = build_index(args.activations_dir, args.dataset_path, tokenizer)
    tokens = load_tokens(args.activations_dir)

    train_idx, heldout_idx = training_inputs(
        function_index, cfg.heldout_frac, cfg.split_seed
    )
    pad = tokenizer.pad_token_id
    train = sample_index(tokens, train_idx, args.positions, pad, generator)
    heldout = sample_index(tokens, heldout_idx, args.heldout_positions, pad, generator)
    index = torch.cat([train, heldout])
    order = index[:, 0].argsort(stable=True)
    index = index[order]
    print(f"{len(train)} train + {len(heldout)} held-out token positions", flush=True)

    d_model = model.config.hidden_size
    out = {
        layer: np.lib.format.open_memmap(
            states_path(args.activations_dir, layer),
            mode="w+",
            dtype=np.float16,
            shape=(len(index), d_model),
        )
        for layer in args.layers
    }
    extract_states(model, tokens, index, args.layers, out, pad)
    for array in out.values():
        array.flush()
    torch.save(index, Path(args.activations_dir) / "token_index.pt")
    write_result(
        Path(args.activations_dir) / "activations.json",
        {
            "layers": args.layers,
            "positions": args.positions,
            "heldout_positions": args.heldout_positions,
            "seed": args.seed,
            "n_train_tokens": len(train),
            "n_heldout_tokens": len(heldout),
            "d_model": d_model,
            "split": {"heldout_frac": cfg.heldout_frac, "split_seed": cfg.split_seed},
        },
    )


if __name__ == "__main__":
    main()
