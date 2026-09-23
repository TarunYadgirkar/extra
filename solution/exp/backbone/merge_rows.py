"""Append DINO-grid columns of extra row sets onto a base row set (same sampled pixels).
Usage: merge_rows.py OUT_TAG BASE_TAG EXTRA_TAG[:grids e.g. 0,1 | :all for every column] ...  (splits: train, train_u, val)"""
import re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from devdata import ROOT

PAT = re.compile(r"^(sm|sr|tk|lda|ldar|selfmean|selfmin|n)(\d)(_\w+)?$")


def grid_of(name):
    m = PAT.match(name)
    return int(m.group(2)) if m else None


def load(split, tag):
    d = np.load(ROOT / "cache" / f"rows_{split}_{tag}.npz")
    return d["X"], d["Y"], d["Q"], [str(n) for n in d["names"]]


def merge(split, out, base, extras):
    X, Y, Q, names = load(split, base)
    ngrid = max(g for g in map(grid_of, names) if g is not None) + 1
    shared = [i for i, n in enumerate(names) if grid_of(n) is None]
    cols, new_names = [X], list(names)
    for spec in extras:
        tag, _, gsel = spec.partition(":")
        Xe, Ye, Qe, ne = load(split, tag)
        assert (Ye == Y).all() and (Qe == Q).all(), f"{tag}: row mismatch"
        if gsel == "all":
            cols.append(Xe); new_names += ne
            continue
        se = [i for i, n in enumerate(ne) if grid_of(n) is None]
        assert np.allclose(Xe[:200000][:, se], X[:200000][:, shared]), f"{tag}: shared cols differ"
        keep = sorted({grid_of(n) for n in ne if grid_of(n) is not None})
        if gsel:
            keep = [int(g) for g in gsel.split(",")]
        for g in keep:
            idx = [i for i, n in enumerate(ne) if grid_of(n) == g]
            cols.append(Xe[:, idx])
            new_names += [PAT.sub(lambda m: f"{m.group(1)}{ngrid}{m.group(3) or ''}", ne[i]) for i in idx]
            ngrid += 1
        del Xe
    np.savez(ROOT / "cache" / f"rows_{split}_{out}.npz", X=np.concatenate(cols, 1), Y=Y, Q=Q, names=np.array(new_names))
    print(split, out, len(new_names), flush=True)


if __name__ == "__main__":
    for split in ("train", "train_u", "val"):
        merge(split, sys.argv[1], sys.argv[2], sys.argv[3:])
