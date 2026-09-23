"""Dev-only: ev_ columns aligned row-for-row with cache/hd_{split}_X.npy (replays build_rows.py sampling).

Usage: build_ev.py SPLIT [SPLIT...]   SPLIT in train (writes train + train_u), val
-> cache/ev_{split}_X.npy, cache/ev_names.npy
"""
import sys, time
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from devdata import examples, domain, ROOT  # noqa: E402
from features import load_gray  # noqa: E402
from build_rows import PER_CLASS, UNIFORM  # noqa: E402
from evfeat import EvContext, ev_query, ev_features_at  # noqa: E402

cv2.setNumThreads(4)
C = ROOT / "cache"


def run(split):
    exs = examples(split)
    by_img = {}
    for e in exs:
        by_img.setdefault(e["image"], []).append(e)
    modes = ["b", "u"] if split == "train" else ["a"]
    rngs = {m: np.random.default_rng(0) for m in modes}
    out = {m: ([], [], []) for m in modes}
    names, t = None, time.time()
    for n, (img, group) in enumerate(by_img.items()):
        ectx = EvContext(load_gray(ROOT / "dataset" / img))
        for e in group:
            d = domain(e)
            idx = np.arange(len(d["ys"]))
            sel = {}
            if split == "train":
                pos, neg = idx[d["pos"]], idx[~d["pos"]]
                sel["b"] = np.concatenate([rngs["b"].choice(pos, min(PER_CLASS, len(pos)), replace=False),
                                           rngs["b"].choice(neg, min(PER_CLASS, len(neg)), replace=False)])
                if e["kind"] == "real":
                    sel["u"] = rngs["u"].choice(idx, min(UNIFORM, len(idx)), replace=False)
            else:
                sel["a"] = idx
            qc = ev_query(ectx, e["query_box"])
            for m, ii in sel.items():
                f, names = ev_features_at(qc, d["ys"][ii].astype(np.float32), d["xs"][ii].astype(np.float32))
                X, Y, Q = out[m]
                X.append(f); Y.append(d["pos"][ii]); Q.append(np.full(len(ii), exs.index(e)))
        print(split, n, len(by_img), f"{time.time()-t:.0f}s", flush=True)
    for m, (X, Y, Q) in out.items():
        s = {"b": "train", "u": "train_u", "a": split}[m]
        Y, Q = np.concatenate(Y), np.concatenate(Q)
        assert (Y == np.load(C / f"hd_{s}_Y.npy")).all() and (Q == np.load(C / f"hd_{s}_Q.npy")).all(), s
        np.save(C / f"ev_{s}_X.npy", np.concatenate(X))
        np.save(C / "ev_names.npy", np.array(names))
        print("saved", s, flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        run(s)
