"""Dev: fit PCA on train DINO-S grids, then cache per-image input stacks and per-query sims + s14 label grids."""
import sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nn_common import (ROOT, S, NPCA, IMG_DIR, Q_DIR, PCA_PATH, _norm, image_input, LiteCtx, query_sims)
sys.path.insert(0, str(ROOT / "solution"))
from devdata import examples, domain  # noqa: E402
from features import load_gray  # noqa: E402


def stem(img):
    return Path(img).stem[:16]


def grids(img):
    s = stem(img)
    return np.load(ROOT / "cache" / "feat_s_1.0" / f"{s}.npy"), np.load(ROOT / "cache" / "feat_s_0.5" / f"{s}.npy")


def fit_pca():
    rng = np.random.default_rng(0)
    imgs = sorted({e["image"] for e in examples("train")})
    out = {}
    for key, idx in (("14", 0), ("28", 1)):
        samp = []
        for img in imgs:
            f = grids(img)[idx]
            flat = f.reshape(-1, f.shape[-1])
            samp.append(_norm(flat[rng.choice(len(flat), 1500, replace=False)]))
        X = np.concatenate(samp)
        mu = X.mean(0)
        ev, V = np.linalg.eigh(np.cov(X - mu, rowvar=False))
        order = np.argsort(ev)[::-1][:NPCA]
        out["mu" + key] = mu.astype(np.float32)
        out["W" + key] = (V[:, order] / np.sqrt(ev[order])).astype(np.float32)
        print(key, "explained", ev[order].sum() / ev.sum(), flush=True)
    np.savez(PCA_PATH, **out)


def labels14(e, gh, gw):
    d = domain(e)
    cell = (d["ys"] // S) * gw + d["xs"] // S
    k = np.bincount(cell, minlength=gh * gw).reshape(gh, gw)
    p = np.bincount(cell[d["pos"]], minlength=gh * gw).reshape(gh, gw)
    return np.stack([k, p], -1).astype(np.float32) / (S * S)


def do_image(args):
    img, group = args
    pca = dict(np.load(PCA_PATH))
    f14, f28 = grids(img)
    gh, gw = f14.shape[:2]
    p = IMG_DIR / f"{stem(img)}.npy"
    if not p.exists():
        np.save(p, image_input(load_gray(ROOT / "dataset" / img), f14, f28, pca))
    ctx = None
    for e in group:
        q = Q_DIR / f"{e['id']}.npz"
        if q.exists():
            continue
        if ctx is None:
            ctx = LiteCtx([(f28, 28.0), (f14, 14.0)])
        sims, qs = query_sims(ctx, e["query_box"], gh, gw)
        np.savez(q, sims=sims, qs=qs, lab=labels14(e, gh, gw).astype(np.float16))
    return img


if __name__ == "__main__":
    IMG_DIR.mkdir(parents=True, exist_ok=True); Q_DIR.mkdir(parents=True, exist_ok=True)
    if not PCA_PATH.exists():
        fit_pca()
    by = {}
    for s in ("train", "val"):
        for e in examples(s):
            by.setdefault(e["image"], []).append(e)
    t = time.time()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as pool:
        for n, img in enumerate(pool.imap_unordered(do_image, list(by.items()))):
            print(n, len(by), img, f"{time.time()-t:.0f}s", flush=True)
