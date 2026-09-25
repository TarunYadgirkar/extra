"""Dev: cache grayscale images as raw uint8 .npy (for mmap crops) and per-query label grids at 14 px and 28 px cells."""
import sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_common import ROOT, GRAY_DIR, LAB_DIR, stem  # noqa: E402
from devdata import examples, domain  # noqa: E402
from features import load_gray  # noqa: E402


def label_grid(d, H, W, c):
    gh, gw = -(-H // c), -(-W // c)
    cell = (d["ys"] // c) * gw + d["xs"] // c
    k = np.bincount(cell, minlength=gh * gw).reshape(gh, gw)
    p = np.bincount(cell[d["pos"]], minlength=gh * gw).reshape(gh, gw)
    return (np.stack([k, p], -1) / (c * c)).astype(np.float16)


def do_image(args):
    img, group = args
    g = GRAY_DIR / f"{stem(group[0])}.npy"
    gray = None
    if not g.exists():
        gray = load_gray(ROOT / "dataset" / img)
        np.save(g, gray)
    for e in group:
        p = LAB_DIR / f"{e['id']}.npz"
        if p.exists():
            continue
        H, W = e["height"], e["width"]
        d = domain(e)
        np.savez(p, l14=label_grid(d, H, W, 14), l28=label_grid(d, H, W, 28))
    return img


if __name__ == "__main__":
    GRAY_DIR.mkdir(parents=True, exist_ok=True); LAB_DIR.mkdir(parents=True, exist_ok=True)
    by = {}
    for s in ("train", "val"):
        for e in examples(s):
            by.setdefault(e["image"], []).append(e)
    t = time.time()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as pool:
        for n, img in enumerate(pool.imap_unordered(do_image, list(by.items()))):
            if n % 20 == 0:
                print(n, len(by), f"{time.time()-t:.0f}s", flush=True)
    print("prep done")
