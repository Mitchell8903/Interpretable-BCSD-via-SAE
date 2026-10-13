"""BinaryCorp Helpers"""

import hashlib
import json
import os
import pickle
import random
import sys
from pathlib import Path

import torch

from jtrans import MAX_TOKENS
from utils import JTRANS_DIR

OPTS = ["O0", "O1", "O2", "O3", "Os"]

INDEX_FILE = "functions.json"
TOKENS_FILE = "tokens.pt"


def key(row):
    return (row["proj"], row["funcname"], row["opt"])


def sort_key(row):
    """key(row) as strings: 12 functions have integer names."""
    return tuple(str(part) for part in key(row))


def load_functions(dataset_path):
    """BinaryCorp functions as dicts of proj / funcname / opt / asm."""
    dataset_path = str(Path(dataset_path).resolve())
    old = os.getcwd()
    sys.path.insert(0, str(JTRANS_DIR))
    os.chdir(JTRANS_DIR)
    from data import load_paired_data

    functions, ebds = load_paired_data(
        datapath=dataset_path,
        filt=None,
        alldata=True,
        convert_jump=True,
        opt=OPTS,
        add_ebd=True,
    )
    os.chdir(old)

    rows = []
    for asm_strings, meta in zip(functions, ebds):
        for opt in OPTS:
            if meta.get(opt) is not None:
                rows.append(
                    {
                        "proj": meta["proj"],
                        "funcname": meta["funcname"],
                        "opt": opt,
                        "asm": asm_strings[meta[opt]],
                    }
                )
    return rows


def load_index(activations_dir):
    """The function index dataset path and the row order."""
    return json.loads((Path(activations_dir) / INDEX_FILE).read_text())


def load_tokens(activations_dir):
    """Cached token ids of every indexed function, (n, MAX_LEN)."""
    return torch.load(Path(activations_dir) / TOKENS_FILE)


def cache_per_function(dataset_path, rows, cache_path, value):
    """key -> value(assembly text) for rows."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())
    wanted = {key(r) for r in rows}
    values = {
        key(row): value(row["asm"])
        for row in load_functions(dataset_path)
        if key(row) in wanted
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pickle.dumps(values))
    return values


def load_asm(index, rows, activations_dir, max_chars):
    """key -> assembly text truncated to max_chars."""
    cache = Path(activations_dir) / f"asm_{max_chars}.pkl"
    return cache_per_function(
        index["dataset_path"], rows, cache, lambda a: a[:max_chars]
    )


def input_key(asm):
    """Hash of the token sequence jTrans actually reads."""
    return hashlib.sha1(" ".join(asm.split()[:MAX_TOKENS]).encode()).hexdigest()


def input_keys(index, activations_dir="data/activations"):
    """key -> input_key for every indexed function."""
    cache = Path(activations_dir) / "input_keys.pkl"
    return cache_per_function(
        index["dataset_path"], index["functions"], cache, input_key
    )


def first_of_each_input(rows, keys):
    """Indices of the first row carrying each distinct model input."""
    seen, keep = set(), []
    for i, r in enumerate(rows):
        if keys[key(r)] not in seen:
            seen.add(keys[key(r)])
            keep.append(i)
    return keep


def split_by_project(functions, heldout_frac, seed):
    """(train, held-out) row indices, with no project on both sides."""
    projects = sorted({f["proj"] for f in functions})
    random.Random(seed).shuffle(projects)
    heldout = set(projects[: int(len(projects) * heldout_frac)])
    train_idx = [i for i, f in enumerate(functions) if f["proj"] not in heldout]
    heldout_idx = [i for i, f in enumerate(functions) if f["proj"] in heldout]
    return train_idx, heldout_idx


def training_inputs(index, heldout_frac, split_seed):
    """(training rows, one per distinct input; held-out rows)."""
    rows = index["functions"]
    train_idx, heldout_idx = split_by_project(rows, heldout_frac, split_seed)
    keys = input_keys(index)
    heldout = {keys[key(rows[i])] for i in heldout_idx}
    distinct = first_of_each_input([rows[i] for i in train_idx], keys)
    train_idx = [
        train_idx[j] for j in distinct if keys[key(rows[train_idx[j]])] not in heldout
    ]
    return train_idx, heldout_idx
