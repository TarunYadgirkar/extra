"""Component-level verifier: label-free evidence per accepted connected component of the post-processed 4px grid.

comp_table(...) -> (F [n_comp, k], names, lab, qcomp). Usable at inference; no labels anywhere.
"""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

SOL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOL))
from pixfeat import DOWN, features_at  # noqa: E402
from texfeat import NR, NA, EPS, BLOB_R, _box_mean, _bc, tex_features_at  # noqa: E402

MAX_PTS = 2000
TEX_GROUPS = ("s0_", "s1_", "qs", "b6", "b16", "qb", "ncc", "q_")
TONE = ("mean4", "std4", "ink4", "bg4", "paper4", "paperfrac4", "mean12", "paper12", "ink12")


def _stride_lab(lab, s, shape):
    """Component label at each cell centre of a stride-s grid with the given shape."""
    gy = np.minimum(((np.arange(shape[0]) + 0.5) * s // DOWN).astype(int), lab.shape[0] - 1)
    gx = np.minimum(((np.arange(shape[1]) + 0.5) * s // DOWN).astype(int), lab.shape[1] - 1)
    return lab[np.ix_(gy, gx)]


def _region_mean(grid, lab_s, n):
    """Mean over each component of a (h, w[, c]) grid; row 0 is background."""
    L = lab_s.ravel()
    cnt = np.bincount(L, minlength=n + 1).astype(np.float64)
    flat = grid.reshape(len(L), -1).astype(np.float64)
    out = np.stack([np.bincount(L, flat[:, j], n + 1) for j in range(flat.shape[1])], 1)
    out = out / np.maximum(cnt, 1)[:, None]
    out[cnt == 0] = np.nan
    return out, cnt


def _spec_scores(d, qd):
    p = d / (d.sum(1, keepdims=True) + EPS)
    sp, sq = np.sqrt(np.maximum(p, 0)).reshape(-1, NR, NA), np.sqrt(np.maximum(qd, 0)).reshape(NR, NA)
    rot = np.stack([(sp * np.roll(sq, k, axis=1)).sum((1, 2)) for k in range(NA)], 1)
    r, qr = p.reshape(-1, NR, NA).sum(2), qd.reshape(NR, NA).sum(1)
    return {"bc": rot[:, 0], "bcrot": rot.max(1), "bcrad": _bc(r, qr), "l1": np.abs(p - qd).sum(1)}


def _query_comp(lab, box):
    x0, y0, x1, y1 = box
    c = np.bincount(lab[y0 // DOWN:(y1 - 1) // DOWN + 1, x0 // DOWN:(x1 - 1) // DOWN + 1].ravel(), minlength=lab.max() + 1)
    c[0] = 0
    return int(c.argmax()) if c.any() else 0


def _geometry(lab, n, h, g_blur, raw, box, t):
    x0, y0, x1, y1 = box
    qa = float((x1 - x0) * (y1 - y0))
    idx = np.arange(1, n + 1)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:] * DOWN * DOWN
    sl = ndi.find_objects(lab)
    bb = np.array([[(s[0].stop - s[0].start) * DOWN, (s[1].stop - s[1].start) * DOWN] for s in sl], float)
    cy, cx = np.array(ndi.center_of_mass(np.ones_like(lab), lab, idx)).reshape(-1, 2).T * DOWN
    qy, qx = (y0 + y1) / 2, (x0 + x1) / 2
    H, W = lab.shape[0] * DOWN, lab.shape[1] * DOWN
    border = np.zeros(n + 1, bool)
    border[np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])] = True
    filled = ndi.mean((g_blur <= t).astype(np.float32), lab, idx)
    order = np.argsort(-area)
    rank = np.empty(n); rank[order] = np.arange(n)
    q = np.array([np.quantile(raw[s][lab[s] == c], (0.1, 0.5, 0.9)) for c, s in zip(idx, sl)]).reshape(-1, 3)
    f = {
        "p_mean": ndi.mean(raw, lab, idx), "p_q10": q[:, 0], "p_q50": q[:, 1], "p_q90": q[:, 2],
        "p_max": ndi.maximum(raw, lab, idx), "p_gt08": ndi.mean((raw > 0.8).astype(np.float32), lab, idx),
        "h_mean": ndi.mean(h, lab, idx), "fillfrac": filled,
        "larea_q": np.log(area / qa), "larea_img": np.log(area / (H * W)), "acc_share": area / area.sum(),
        "rank_area": rank, "n_comp": np.full(n, np.log(n)), "compact": area / (bb[:, 0] * bb[:, 1]),
        "aspect": np.log(bb.max(1) / bb.min(1)), "border": border[1:].astype(float),
        "dist_diag": np.hypot(cy - qy, cx - qx) / np.hypot(H, W), "dist_q": np.log1p(np.hypot(cy - qy, cx - qx) / np.sqrt(qa)),
    }
    return f


def _region(ctx, tctx, cache, qc, lab, n, box):
    """Region-level descriptors (component-pooled first, compared second)."""
    f = {}
    for i, ((desc, loge), st) in enumerate(tctx.spec):
        ls = _stride_lab(lab, st, loge.shape)
        d, _ = _region_mean(desc, ls, n)
        e, _ = _region_mean(loge, ls, n)
        qd = _box_mean(desc, box, st); qd = qd / (qd.sum() + EPS)
        for k, v in _spec_scores(d[1:], qd).items():
            f[f"r_s{i}_{k}"] = v
        f[f"r_s{i}_ed"] = e[1:, 0] - float(_box_mean(loge, box, st))
    ls = _stride_lab(lab, DOWN, tctx.blob["cnt6"].shape[:2])
    for r in BLOB_R:
        for kind in ("cnt", "area"):
            m = tctx.blob[f"{kind}{r}"]
            d, _ = _region_mean(m, ls, n)
            qv = _box_mean(m, box, DOWN)
            ds, qs = d[1:].sum(1), qv.sum()
            f[f"r_b{r}_bc{kind}"] = _bc(d[1:] / (ds[:, None] + EPS), qv / (qs + EPS))
            f[f"r_b{r}_l{kind}"] = np.log((ds + 1e-3) / (qs + 1e-3))
    for i, ((grid, s), p) in enumerate(zip(ctx.dino, cache["protos"])):
        ls = _stride_lab(lab, s, grid.shape[:2])
        d, _ = _region_mean(grid, ls, n)
        m = d[1:]
        mn = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-6)
        f[f"r_d{i}_cos"] = mn @ p["mean"]
        f[f"r_d{i}_cosrob"] = mn @ p["rob"]
        wv, lo, hi = p["lda"]
        f[f"r_d{i}_lda"] = (m @ wv - lo) / (hi - lo + 1e-6)
        f[f"r_d{i}_coh"] = np.linalg.norm(m, axis=1)
    ls = _stride_lab(lab, DOWN, ctx.tone["mean4"].shape)
    for k in TONE:
        med = ndi.median(ctx.tone[k], ls, np.arange(1, n + 1))
        f[f"r_t_{k}_ad"] = np.abs(np.asarray(med) - cache["q"][k])
    return f


def _pooled(ctx, tctx, cache, qc, lab, n, box, rng):
    """Mean of the head's pixel feature columns over (subsampled) component cells and over the query box."""
    ys_all, xs_all = np.nonzero(lab)
    L = lab[ys_all, xs_all]
    sel = []
    for c in range(1, n + 1):
        ii = np.nonzero(L == c)[0]
        sel.append(ii if len(ii) <= MAX_PTS else rng.choice(ii, MAX_PTS, replace=False))
    x0, y0, x1, y1 = box
    qy, qx = np.mgrid[y0 + DOWN // 2:y1:DOWN, x0 + DOWN // 2:x1:DOWN]
    ii = np.concatenate(sel)
    ys = np.r_[ys_all[ii] * DOWN + DOWN // 2, qy.ravel()].astype(np.float32)
    xs = np.r_[xs_all[ii] * DOWN + DOWN // 2, qx.ravel()].astype(np.float32)
    fa, fn = features_at(ctx, box, ys, xs, cache)
    ta, tn = tex_features_at(tctx, qc, ys, xs)
    keep = [j for j, k in enumerate(tn) if k.startswith(TEX_GROUPS)]
    X = np.concatenate([fa, ta[:, keep]], 1)
    names = fn + [tn[j] for j in keep]
    grp = np.r_[L[ii], np.zeros(len(qy.ravel()), int)]
    cnt = np.bincount(grp, minlength=n + 1)
    M = np.stack([np.bincount(grp, np.nan_to_num(X[:, j]), n + 1) for j in range(X.shape[1])], 1) / np.maximum(cnt, 1)[:, None]
    return M[1:], M[0], names


def comp_table(ctx, tctx, qc, box, raw, t=0.5, seed=0, pooled=True):
    """raw: head prob grid (4px). Returns features per component of the post-processed mask, names, lab, query comp id, h."""
    import cv2
    g_blur = cv2.GaussianBlur(raw, (0, 0), 14 / DOWN)
    lab0, n0 = ndi.label(g_blur <= t)
    fill = np.ones(n0 + 1, bool); fill[0] = False
    fill[np.unique(np.r_[lab0[0], lab0[-1], lab0[:, 0], lab0[:, -1]])] = False
    h = np.where(fill[lab0], t + 0.01, g_blur)
    lab, n = ndi.label(h > t)
    qcomp = _query_comp(lab, box)
    if n == 0:
        return np.zeros((0, 0), np.float32), [], lab, qcomp, h
    cache = {}
    features_at(ctx, box, np.zeros(1, np.float32), np.zeros(1, np.float32), cache)
    f = _geometry(lab, n, h, g_blur, raw, box, t)
    f["is_q"] = (np.arange(1, n + 1) == qcomp).astype(float)
    x0, y0, x1, y1 = box
    qp = float(raw[y0 // DOWN:(y1 - 1) // DOWN + 1, x0 // DOWN:(x1 - 1) // DOWN + 1].mean())
    f["q_p_mean"] = np.full(n, qp); f["d_p_mean"] = f["p_mean"] - qp
    f.update(_region(ctx, tctx, cache, qc, lab, n, box))
    names = list(f)
    F = np.stack([np.asarray(f[k], np.float64) for k in names], 1)
    if pooled:
        M, A, pn = _pooled(ctx, tctx, cache, qc, lab, n, box, np.random.default_rng(seed))
        F = np.concatenate([F, M, M - A[None]], 1)
        names += ["m_" + k for k in pn] + ["dm_" + k for k in pn]
    return F.astype(np.float32), names, lab, qcomp, h
