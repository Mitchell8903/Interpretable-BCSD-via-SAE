"""Build a BinaryCorp-format dataset from unstripped ELF binaries with IDA.

    python src/build_dataset.py --input path/to/binaries --output data/MyCorp
"""

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from pathlib import Path

import sys

import networkx as nx

from binarycorp import OPTS
from utils import JTRANS_DIR

sys.path.insert(0, str(JTRANS_DIR / "datautils"))
from util.base import Binarybase  # noqa: E402

IDA = "/mnt/c/Program Files/IDA Essential 9.4/idat.exe"
IDA_SCRIPT = Path(__file__).resolve().parent / "ida_extract.py"


def is_elf(path):
    with open(path, "rb") as f:
        return f.read(4) == b"\x7fELF"


def find_binaries(root):
    """(project, opt, path) for every ELF file under root/<project>/<opt>/."""
    found = []
    for project in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for opt_dir in sorted(p for p in project.iterdir() if p.is_dir()):
            if opt_dir.name not in OPTS:
                print(f"skipping {opt_dir}: not one of {OPTS}", flush=True)
                continue
            for path in sorted(opt_dir.rglob("*")):
                if path.is_file() and is_elf(path):
                    found.append((project.name, opt_dir.name, path))
    return found


def addr2name(path):
    """Symbol address -> name (-1 if none), from jTrans util/base.py itself."""
    return Binarybase(str(path)).addr2name


def build_cfg(blocks):
    """networkx DiGraph with the node and edge insertion order of process.py."""
    graph = nx.DiGraph()
    for block in blocks:
        graph.add_node(block["start"], asm=block["asm"], raw=bytes.fromhex(block["raw"]))
        for pred in block["preds"]:
            graph.add_edge(pred, block["start"])
        for succ in block["succs"]:
            graph.add_edge(block["start"], succ)
    return graph


def to_host_path(path, windows):
    if not windows:
        return str(path)
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()


def run_ida(ida, path, work_dir, timeout, keep_idb):
    """BinaryCorp function dict of one binary (name -> 5-element record)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    stripped = work_dir / (path.name + ".strip")
    subprocess.run(["strip", "-s", str(path), "-o", str(stripped)], check=True)
    names = addr2name(path)
    (work_dir / (stripped.name + ".syms.json")).write_text(json.dumps(list(names)))

    windows = ida.endswith(".exe")
    out = work_dir / (stripped.name + ".extract.json")
    out.unlink(missing_ok=True)
    idb = work_dir / (path.name + ".i64")
    log = work_dir / (path.name + ".ida.log")
    cmd = [
        ida,
        "-A",
        "-c",
        f"-L{to_host_path(log, windows)}",
        f"-S{to_host_path(IDA_SCRIPT, windows)}",
        f"-o{to_host_path(idb, windows)}",
        to_host_path(stripped, windows),
    ]
    subprocess.run(cmd, cwd=work_dir, timeout=timeout, capture_output=True)
    if not out.exists():
        raise RuntimeError(f"IDA wrote no output, see {log}")

    functions = {}
    for f in json.loads(out.read_text()):
        # unnamed functions all land on -1, as in BinaryCorp
        functions[names[f["addr"]]] = [
            f["addr"],
            f["asm"],
            bytes.fromhex(f["raw"]),
            build_cfg(f["cfg"]),
            None,  # BinaryAI feature; the binaryai package is not used here
        ]

    out.unlink()
    stripped.unlink()
    (work_dir / (stripped.name + ".syms.json")).unlink()
    if not keep_idb:
        for leftover in work_dir.glob(path.name + ".i64*"):
            leftover.unlink()
    return functions


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def write_pickle(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(value))
    os.replace(tmp, path)


def pair(proj_dir):
    """saved_index.pkl: functions present in every binary of proj_dir."""
    pkls = [
        pickle.loads(p.read_bytes()) for p in sorted(proj_dir.glob("*_extract.pkl"))
    ]
    common = reduce(lambda a, b: a & b, (set(p) for p in pkls))
    write_pickle(
        proj_dir / "saved_index.pkl",
        {name: [p[name] for p in pkls] for name in common},
    )
    return len(common)


def main():
    parser = argparse.ArgumentParser(description="Build a BinaryCorp-format dataset")
    parser.add_argument("--input", required=True, help="<project>/<opt>/<binary> tree")
    parser.add_argument("--output", required=True)
    parser.add_argument("--work_dir", help="IDA scratch space, default <output>/.ida_work")
    parser.add_argument("--ida", default=IDA, help=f"idat binary, default {IDA}")
    parser.add_argument("--jobs", type=int, default=4, help="parallel IDA instances")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per binary")
    parser.add_argument("--keep_idb", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    work_dir = Path(args.work_dir).resolve() if args.work_dir else output / ".ida_work"
    if args.ida.endswith(".exe") and not str(work_dir).startswith("/mnt/"):
        parser.error("Windows IDA needs --work_dir on a Windows drive (/mnt/...)")

    jobs = []
    for project, opt, path in find_binaries(args.input):
        proj = f"{project}-{path.name}"
        dst = output / proj / f"{proj}-{opt}-{md5(path)}_extract.pkl"
        if dst.exists():
            print(f"have {dst.name}, skipping", flush=True)
        else:
            jobs.append((path, dst))
    print(f"{len(jobs)} binaries to extract", flush=True)

    failed = []
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = {
            pool.submit(
                run_ida, args.ida, path, work_dir / dst.stem, args.timeout, args.keep_idb
            ): (path, dst)
            for path, dst in jobs
        }
        for i, future in enumerate(as_completed(futures), 1):
            path, dst = futures[future]
            try:
                functions = future.result()
            except Exception as e:
                failed.append(path)
                print(f"[{i}/{len(jobs)}] FAILED {path}: {e}", flush=True)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            write_pickle(dst, functions)
            print(f"[{i}/{len(jobs)}] {dst.name}: {len(functions)} functions", flush=True)

    for proj_dir in sorted(p for p in output.iterdir() if p.is_dir() and p != work_dir):
        n_opts = len(list(proj_dir.glob("*_extract.pkl")))
        print(f"{proj_dir.name}: {n_opts} opt levels, {pair(proj_dir)} paired functions")
    if not failed and not args.keep_idb:
        shutil.rmtree(work_dir, ignore_errors=True)
    if failed:
        raise SystemExit(f"{len(failed)} binaries failed: {[str(p) for p in failed]}")


if __name__ == "__main__":
    main()
