"""Dev: per-component verifier tables for train real queries (OOF grids) and val (deployed-head grids).
python build.py [train|val] -> cache/vf_comp/{id}.npz"""
import sys, time
from pathlib import Path
import numpy as np
from vbase import CACHE, ROOT, queries, load_raw, dom_cells, Dom, DOWN, sample_grid
from verifier import comp_table
from features import load_gray
from pixfeat import ImageContext
from texfeat import TexContext, tex_query

OUT = CACHE / "vf_comp"; OUT.mkdir(exist_ok=True)
SCALES = [("feat_s_0.5", 28.0), ("feat_s_1.0", 14.0)]


def labels(e, h, lab, n):
    D = Dom(e)
    m = sample_grid(h, DOWN, D.ys, D.xs) > 0.5
    L = lab[dom_cells(D.ys, D.xs, h.shape)]; L[~m] = 0
    pos = np.bincount(L, D.y, n + 1)[1:]; tot = np.bincount(L, minlength=n + 1)[1:]
    tp = int((m & D.y).sum()); fp = int((m & ~D.y).sum()); fn = int((~m & D.y).sum())
    return pos, tot, tp, fp, fn


for split in sys.argv[1:]:
    exs, qs = queries(split)
    by = {}
    for i, e, f in qs:
        by.setdefault(e["image"], []).append(e)
    t0 = time.time()
    for k, (img, group) in enumerate(by.items()):
        todo = [e for e in group if not (OUT / f"{e['id']}.npz").exists()]
        if not todo:
            continue
        stem = Path(img).stem[:16]
        gray = load_gray(ROOT / "dataset" / img)
        ctx = ImageContext(gray, [(np.load(CACHE / d / f"{stem}.npy"), s) for d, s in SCALES])
        tctx = TexContext(gray)
        for e in todo:
            tq = time.time()
            qc = tex_query(tctx, e["query_box"])
            F, names, lab, qcomp, h = comp_table(ctx, tctx, qc, e["query_box"], load_raw(e))
            n = len(F)
            pos, tot, tp, fp, fn = labels(e, h, lab, n)
            np.savez_compressed(OUT / f"{e['id']}.npz", F=F, names=np.array(names), pos=pos, tot=tot, qcomp=qcomp, base=np.array([tp, fp, fn]), sec=time.time() - tq)
        print(split, k + 1, len(by), f"{time.time()-t0:.0f}s", flush=True)
