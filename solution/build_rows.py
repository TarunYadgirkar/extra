"""Dev-only: build feature rows for train (subsampled) and val (all known pixels)."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from devdata import examples, domain, ROOT
from features import load_gray
from pixfeat import ImageContext, features_at

SCALES = [("feat_s_0.5", 28.0), ("feat_s_1.0", 14.0)]
PER_CLASS = 2500


def run(split, tag):
    rng = np.random.default_rng(0)
    exs = examples(split)
    by_img = {}
    for e in exs:
        by_img.setdefault(e["image"], []).append(e)
    X, Y, Q, names = [], [], [], None
    t = time.time()
    for n, (img, group) in enumerate(by_img.items()):
        stem = Path(img).stem[:16]
        dino = [(np.load(ROOT / "cache" / d / f"{stem}.npy"), s) for d, s in SCALES]
        ctx = ImageContext(load_gray(ROOT / "dataset" / img), dino)
        for e in group:
            d = domain(e)
            idx = np.arange(len(d["ys"]))
            if split == "train":
                pos, neg = idx[d["pos"]], idx[~d["pos"]]
                idx = np.concatenate([rng.choice(pos, min(PER_CLASS, len(pos)), replace=False),
                                      rng.choice(neg, min(PER_CLASS, len(neg)), replace=False)])
            f, names = features_at(ctx, e["query_box"], d["ys"][idx], d["xs"][idx])
            X.append(f); Y.append(d["pos"][idx]); Q.append(np.full(len(idx), exs.index(e)))
        if n % 20 == 0:
            print(split, n, len(by_img), f"{time.time()-t:.0f}s", flush=True)
    np.savez(ROOT / "cache" / f"rows_{split}_{tag}.npz", X=np.concatenate(X), Y=np.concatenate(Y), Q=np.concatenate(Q), names=np.array(names))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
