"""Dev-only: texture feature columns aligned row-for-row with solution/build_rows.py output.

Usage: build_tx.py SPLIT TAG [--extra]  -> cache/rows_{split}[_u]_{TAG}.npz holding only the texture columns.
Sampling replays build_rows.py's rng sequence exactly; Y is saved so alignment can be checked.
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
from txfeat import TexContext, tex_query, tex_features_at  # noqa: E402

cv2.setNumThreads(4)


def run(split, tag, extra=False):
    exs = examples(split)
    by_img = {}
    for e in exs:
        by_img.setdefault(e["image"], []).append(e)
    modes = ["b", "u"] if split == "train" else ["a"]
    rngs = {m: np.random.default_rng(0) for m in modes}
    out = {m: ([], [], []) for m in modes}
    names, t = None, time.time()
    for n, (img, group) in enumerate(by_img.items()):
        tctx = TexContext(load_gray(ROOT / "dataset" / img))
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
            qc = tex_query(tctx, e["query_box"], extra)
            for m, ii in sel.items():
                f, names = tex_features_at(tctx, qc, d["ys"][ii].astype(np.float32), d["xs"][ii].astype(np.float32))
                X, Y, Q = out[m]
                X.append(f); Y.append(d["pos"][ii]); Q.append(np.full(len(ii), exs.index(e)))
        print(split, n, len(by_img), f"{time.time()-t:.0f}s", flush=True)
    for m, (X, Y, Q) in out.items():
        suffix = "_u" if m == "u" else ""
        np.savez(ROOT / "cache" / f"rows_{split}{suffix}_{tag}.npz", X=np.concatenate(X), Y=np.concatenate(Y),
                 Q=np.concatenate(Q), names=np.array(names))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], "--extra" in sys.argv)
