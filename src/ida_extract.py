"""IDAPython side of build_dataset.py; runs inside IDA, not the conda env.

    idat -A -c -S ida_extract.py -o x.i64 x.strip

Mirrors jTrans datautils/process.py: same function filter, GetDisasm text and
basic-block walk. Reads the unstripped binary's symbol addresses from
x.strip.syms.json and writes plain JSON to x.strip.extract.json, so it needs
nothing outside IDA's own Python. build_dataset.py turns that into the
BinaryCorp pickle (networkx CFG and all).
"""

import json
import os

import ida_auto
import ida_pro
import idaapi
import idautils
import idc

SKIP_SEGMENTS = [".plt", "extern", ".init", ".fini"]


def block_attr(block, func_addr_set):
    """(asm, raw bytes) of one basic block, None if outside the function."""
    if block.start_ea not in func_addr_set:
        return None
    asm, raw = [], b""
    curr_addr = block.start_ea
    while curr_addr <= block.end_ea:
        asm.append(idc.GetDisasm(curr_addr))
        raw += idc.get_bytes(curr_addr, idc.get_item_size(curr_addr))
        curr_addr = idc.next_head(curr_addr, block.end_ea)
    return asm, raw


def get_cfg(func):
    """Blocks in flowchart order, each with its in-function preds and succs."""
    flowchart = idaapi.FlowChart(idaapi.get_func(func), flags=idaapi.FC_PREDS)
    func_addr_set = set(idautils.FuncItems(func))
    blocks = []
    for block in flowchart:
        attr = block_attr(block, func_addr_set)
        if attr is None:
            continue
        blocks.append(
            {
                "start": block.start_ea,
                "asm": attr[0],
                "raw": attr[1].hex(),
                "preds": [p.start_ea for p in block.preds() if p.start_ea in func_addr_set],
                "succs": [s.start_ea for s in block.succs() if s.start_ea in func_addr_set],
            }
        )
    return blocks


def extract(addrs):
    """One record per function IDA knows, in idautils.Functions() order."""
    for addr in addrs:
        # in case some functions' instructions are not recognised by IDA
        idc.create_insn(addr)
        idc.add_func(addr)

    for func in idautils.Functions():
        if idc.get_segm_name(func) in SKIP_SEGMENTS:
            continue
        items = list(idautils.FuncItems(func))
        yield {
            "addr": func,
            "asm": [idc.GetDisasm(ea) for ea in items],
            "raw": b"".join(idc.get_bytes(ea, idc.get_item_size(ea)) for ea in items).hex(),
            "cfg": get_cfg(func),
        }


def main():
    path = idc.get_input_file_path()
    with open(path + ".syms.json") as f:
        addrs = json.load(f)
    ida_auto.auto_wait()
    functions = list(extract(addrs))
    with open(path + ".extract.json.tmp", "w") as f:
        json.dump(functions, f)
    # rename last so a half-written file never looks finished
    os.replace(path + ".extract.json.tmp", path + ".extract.json")


main()
ida_pro.qexit(0)
