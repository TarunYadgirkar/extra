"""Dev-only: materialize tx_v1f feature matrices (v3 cols + tx_v2 groups used by infer.py) as .npy memmaps.

Writes cache/hd_{train,train_u,val}_{X,Y,Q}.npy and cache/hd_names.npy.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from devdata import ROOT  # noqa: E402

GROUPS = ("s0_", "s1_", "qs", "b6", "b16", "qb", "ncc", "q_")
C = ROOT / "cache"


def run(split):
    a = np.load(C / f"rows_{split}_v3.npz")
    Y, Q, names = a["Y"], a["Q"], list(a["names"])
    X = a["X"]
    out = np.lib.format.open_memmap(C / f"hd_{split}_X.npy", "w+", np.float32, (len(Y), 162))
    out[:, :X.shape[1]] = X
    nv = X.shape[1]
    del X, a
    t = np.load(C / f"rows_{split}_tx_v2.npz")
    assert (t["Y"] == Y).all() and (t["Q"] == Q).all()
    tn = list(t["names"])
    keep = [i for i, n in enumerate(tn) if n.startswith(GROUPS)]
    assert nv + len(keep) == 162
    out[:, nv:] = t["X"][:, keep]
    out.flush()
    np.save(C / f"hd_{split}_Y.npy", Y); np.save(C / f"hd_{split}_Q.npy", Q.astype(np.int32))
    np.save(C / "hd_names.npy", np.array(names + [tn[i] for i in keep]))
    print(split, out.shape, flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        run(s)
