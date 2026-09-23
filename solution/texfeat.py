"""Query-conditioned hand-crafted texture features (CPU only).

Three families, each an image-level map compared against the query's own value:
  spec: windowed power spectrum in log-polar bins (period x orientation)
  blob: adaptive-ink connected-component size-class densities
  ncc:  normalized cross-correlation of query crops (plus rot90/flip) at 0.5x
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import scipy.fft as sfft

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pixfeat import sample_grid, _cells  # noqa: E402

DOWN = 4
SPEC = ((1.0, 64, 16), (0.5, 64, 16))  # (image scale, window px at that scale, native stride)
P_EDGES = np.array([2, 2.83, 4, 5.66, 8, 11.3, 16, 22.6, 32, 64.01])
NR, NA = len(P_EDGES) - 1, 8
LOGC = np.log(np.sqrt(P_EDGES[:-1] * P_EDGES[1:]))
AREA_EDGES = np.array([3, 8, 20, 50, 150, 500])
NCLS = len(AREA_EDGES) + 1
BLOB_R = (6, 16)
SHAPE_AREA = np.array([8, 50, 300])
NCC_SCALE = 0.5
EPS = 1e-6


def _bin_matrix(W):
    fy = np.fft.fftfreq(W)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    per = 1.0 / np.maximum(f, 1e-9)
    rb = np.digitize(per, P_EDGES) - 1
    ab = np.minimum((np.arctan2(fy, fx) % np.pi / np.pi * NA).astype(int), NA - 1)
    wt = np.where((fx == 0) | (fx == 0.5), 0.5, 1.0) * np.ones_like(f)
    ok = (f > 0) & (rb >= 0) & (rb < NR)
    M = np.zeros((f.size, NR * NA), np.float32)
    src = np.nonzero(ok.ravel())[0]
    M[src, (rb * NA + ab).ravel()[src]] = wt.ravel()[src]
    return M


def spec_grid(gray, scale, W, stride):
    """Normalized log-polar power spectrum per window + log energy, on a native-stride grid."""
    img = gray.astype(np.float32) / 255.0
    Hn, Wn = gray.shape
    if scale != 1.0:
        img = cv2.resize(img, (round(Wn * scale), round(Hn * scale)), interpolation=cv2.INTER_AREA)
    s = int(round(stride * scale))
    gh, gw = -(-Hn // stride), -(-Wn // stride)
    off = (W - s) // 2
    pad = np.ones((gh * s + 2 * W, gw * s + 2 * W), np.float32)
    pad[W:W + img.shape[0], W:W + img.shape[1]] = img[:gh * s, :gw * s]
    base = W - off
    han = np.outer(np.hanning(W), np.hanning(W)).astype(np.float32)
    M = _bin_matrix(W)
    desc = np.empty((gh, gw, NR * NA), np.float32)
    loge = np.empty((gh, gw), np.float32)
    band = max(1, 6000 // gw)
    for r0 in range(0, gh, band):
        r1 = min(gh, r0 + band)
        sl = pad[base + r0 * s: base + (r1 - 1) * s + W, base: base + (gw - 1) * s + W]
        win = np.lib.stride_tricks.sliding_window_view(sl, (W, W))[::s, ::s]
        win = win.reshape(-1, W, W)
        win = (win - win.mean((1, 2), keepdims=True)) * han
        pw = np.abs(sfft.rfft2(win, workers=4)) ** 2
        b = pw.reshape(len(win), -1) @ M
        tot = b.sum(1)
        desc[r0:r1] = (b / (tot[:, None] + EPS)).reshape(r1 - r0, gw, -1)
        loge[r0:r1] = np.log(tot + 1e-4).reshape(r1 - r0, gw)
    return desc, loge


def blob_maps(gray):
    """Per-class component count and ink-area densities on the DOWN grid, at two radii."""
    bg = cv2.dilate(gray, np.ones((9, 9), np.uint8))
    ink = ((bg.astype(np.int16) - gray) > 40).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink, connectivity=8)
    area = st[1:, 4]
    cls = np.digitize(area, AREA_EDGES)
    H, W = gray.shape
    gh, gw = H // DOWN, W // DOWN
    cnt = np.zeros((gh, gw, NCLS), np.float32)
    cy = np.clip((cen[1:, 1] // DOWN).astype(int), 0, gh - 1)
    cx = np.clip((cen[1:, 0] // DOWN).astype(int), 0, gw - 1)
    np.add.at(cnt, (cy, cx, cls), 1.0)
    bw, bh = st[1:, 2], st[1:, 3]
    elong = np.maximum(bw, bh) / np.maximum(1, np.minimum(bw, bh))
    shp = np.digitize(area, SHAPE_AREA) * 2 + (elong >= 2.5)
    sc = np.zeros((gh, gw, 2 * (len(SHAPE_AREA) + 1)), np.float32)
    np.add.at(sc, (cy, cx, shp), 1.0)
    lut = np.concatenate([[255], cls]).astype(np.uint8)
    pc = lut[lab]
    del lab
    ar = np.stack([cv2.resize((pc == c).astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
                   for c in range(NCLS)], -1)
    out = {}
    for r in BLOB_R:
        k = 2 * r + 1
        out[f"cnt{r}"] = cv2.blur(cnt, (k, k))
        out[f"area{r}"] = cv2.blur(ar, (k, k))
        out[f"shape{r}"] = cv2.blur(sc, (k, k))
    return out


class TexContext:
    def __init__(self, gray):
        self.shape = gray.shape
        self.spec = [(spec_grid(gray, sc, W, st), st) for sc, W, st in SPEC]
        self.blob = blob_maps(gray)
        h, w = gray.shape
        self.full = gray.astype(np.float32) / 255.0
        self.half = cv2.resize(gray.astype(np.float32) / 255.0, (round(w * NCC_SCALE), round(h * NCC_SCALE)),
                               interpolation=cv2.INTER_AREA)


def _box_mean(grid, box, stride):
    iy, ix = _cells(box, stride, *grid.shape[:2])
    return grid[np.ix_(iy, ix)].reshape(-1, *grid.shape[2:]).mean(0)


def _bc(a, b):
    return np.sqrt(np.maximum(a, 0)) @ np.sqrt(np.maximum(b, 0))


def spec_compare(desc, loge, qd, qe):
    """Per-cell comparison maps between local spectra (gh,gw,72) and query spectrum (72,)."""
    gh, gw = loge.shape
    p = desc.reshape(-1, NR, NA)
    q = qd.reshape(NR, NA)
    sp, sq = np.sqrt(np.maximum(p, 0)), np.sqrt(np.maximum(q, 0))
    rot = np.stack([(sp * np.roll(sq, k, axis=1)).sum((1, 2)) for k in range(NA)], 1)
    r, qr = p.sum(2), q.sum(1)
    a, qa = p.sum(1), q.sum(0)
    shift = rot.argmax(1)
    m = {
        "bc": rot[:, 0],
        "l1": np.abs(p - q).sum((1, 2)),
        "bcrot": rot.max(1),
        "rotd": np.minimum(shift, NA - shift).astype(np.float32),
        "bcrad": _bc(r, qr),
        "l1rad": np.abs(r - qr).sum(1),
        "bcang": _bc(a, qa),
        "centd": r @ LOGC - qr @ LOGC,
        "ed": loge.ravel() - qe,
        "peakd": p.reshape(len(p), -1).max(1) - q.max(),
    }
    return {k: v.reshape(gh, gw).astype(np.float32) for k, v in m.items()}


def blob_compare(blob, box):
    out, qc = {}, {}
    for r in BLOB_R:
        c, a = blob[f"cnt{r}"], blob[f"area{r}"]
        q_c, q_a = _box_mean(c, box, DOWN), _box_mean(a, box, DOWN)
        cs, qs = c.sum(-1), q_c.sum()
        as_, qas = a.sum(-1), q_a.sum()
        k = int(q_a.argmax())
        out[f"b{r}_bccnt"] = _bc(c / (cs[..., None] + EPS), q_c / (qs + EPS))
        out[f"b{r}_bcarea"] = _bc(a / (as_[..., None] + EPS), q_a / (qas + EPS))
        out[f"b{r}_lcnt"] = np.log((cs + 1e-3) / (qs + 1e-3))
        out[f"b{r}_larea"] = np.log((as_ + 1e-3) / (qas + 1e-3))
        out[f"b{r}_ldom"] = np.log((a[..., k] + 1e-3) / (q_a[k] + 1e-3))
        qc[f"qb{r}_lcnt"] = float(np.log(qs + 1e-3))
        qc[f"qb{r}_dom"] = float(k)
    return out, qc


def _ncc_map(img, t, pool):
    th, tw = t.shape
    res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
    res = np.nan_to_num(np.clip(res, -1, 1))
    full = np.zeros(img.shape, np.float32)
    full[th // 2: th // 2 + res.shape[0], tw // 2: tw // 2 + res.shape[1]] = res
    return cv2.dilate(full, np.ones((pool, pool), np.uint8))


def _templates(half, box):
    x0, y0, x1, y1 = [int(round(v * NCC_SCALE)) for v in box]
    crop = half[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
    h, w = crop.shape
    ch, cw = min(h, 48), min(w, 48)
    cy, cx = (h - ch) // 2, (w - cw) // 2
    t0 = crop[cy:cy + ch, cx:cx + cw]
    tiles, centers = [], []
    if min(h, w) >= 24:
        t = int(np.clip(min(h, w) * 0.6, 10, 40))
        for oy, ox in ((0, 0), (0, w - t), (h - t, 0), (h - t, w - t)):
            tiles.append(crop[oy:oy + t, ox:ox + t])
            centers.append((y0 + oy + t // 2, x0 + ox + t // 2))
    return t0, tiles, centers


def ncc_query_maps(tctx, box):
    """NCC maps on the DOWN grid; empty dict entries become NaN when the query crop is flat."""
    half = tctx.half
    gh, gw = tctx.shape[0] // DOWN, tctx.shape[1] // DOWN
    t0, tiles, centers = _templates(half, box)
    names = ["ncc_c", "ncc_rot", "ncc_best", "ncc_max", "ncc_med", "ncc_cblur"]
    q = {"q_nccself": np.nan, "q_tstd": float(t0.std())}
    if t0.std() < 3 / 255 or min(t0.shape) < 4:
        return {k: None for k in names}, q
    pool = max(3, min(t0.shape) // 2) | 1
    down = lambda m: cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA)
    mc = _ncc_map(half, t0, pool)
    mrot = np.maximum(_ncc_map(half, np.ascontiguousarray(np.rot90(t0)), pool),
                      _ncc_map(half, np.ascontiguousarray(t0[:, ::-1]), pool))
    out = {"ncc_c": down(mc), "ncc_rot": down(mrot)}
    out["ncc_best"] = np.maximum(out["ncc_c"], out["ncc_rot"])
    ok = [i for i, t in enumerate(tiles) if t.std() >= 3 / 255]
    tm = [_ncc_map(half, tiles[i], max(3, tiles[i].shape[0] // 2) | 1) for i in ok]
    centers = [centers[i] for i in ok]
    if len(tm) >= 2:
        st = np.stack([down(m) for m in tm])
        out["ncc_max"] = np.maximum(st.max(0), out["ncc_c"])
        out["ncc_med"] = np.median(st, 0)
        selfs = [tm[j][centers[i]] for i in range(len(tm)) for j in range(len(tm)) if i != j]
        q["q_nccself"] = float(np.mean(selfs))
    else:
        out["ncc_max"], out["ncc_med"] = out["ncc_c"], out["ncc_c"]
    out["ncc_cblur"] = cv2.blur(out["ncc_c"], (13, 13))
    return out, q


def tex_query(tctx, box, extra=False):
    maps, consts = [], {}
    for i, ((desc, loge), st) in enumerate(tctx.spec):
        qd = _box_mean(desc, box, st); qd = qd / (qd.sum() + EPS)
        qe = float(_box_mean(loge, box, st))
        m = spec_compare(desc, loge, qd, qe)
        maps += [(f"s{i}_{k}", v, st) for k, v in m.items()]
        consts[f"qs{i}_e"] = qe
        consts[f"qs{i}_peak"] = float(qd.max())
        consts[f"qs{i}_cent"] = float(qd.reshape(NR, NA).sum(1) @ LOGC)
    bm, bq = blob_compare(tctx.blob, box)
    maps += [(k, v, DOWN) for k, v in bm.items()]
    consts.update(bq)
    nm, nq = ncc_query_maps(tctx, box)
    for k, v in nm.items():
        maps.append((k, v, DOWN))
    consts.update(nq)
    quant = {k: np.quantile(v, np.linspace(0, 1, 101)) for k, v, _ in maps if k in ("ncc_c", "ncc_best") and v is not None}
    qc = {"maps": maps, "consts": consts, "quant": quant}
    if extra:
        qc["extra"] = tex_extra(tctx, qc, box)
    return qc


def _blur(m, r):
    return cv2.blur(m, (2 * r + 1, 2 * r + 1))


def tex_extra(tctx, qc, box):
    """v2 columns (prefix x_): region-pooled versions of the strongest maps + native-scale NCC."""
    maps = {k: (v, st) for k, v, st in qc["maps"]}
    out = []
    for k, radii in (("ncc_best", (4, 16)), ("ncc_max", (16,)), ("s0_bcrot", (2, 4)), ("s1_bcrot", (4,)),
                     ("s0_bcrad", (4,)), ("b16_bccnt", (16,)), ("b6_ldom", (16,))):
        v, st = maps[k]
        for r in radii:
            out.append((f"x_{k}_b{r}", None if v is None else _blur(v, r), st))
    for r in BLOB_R:
        c = tctx.blob[f"shape{r}"]
        q = _box_mean(c, box, DOWN)
        cs, qs = c.sum(-1), q.sum()
        k = int(q.argmax())
        out.append((f"x_sb{r}_bc", _bc(c / (cs[..., None] + EPS), q / (qs + EPS)), DOWN))
        out.append((f"x_sb{r}_ldom", np.log((c[..., k] + 1e-3) / (q[k] + 1e-3)), DOWN))
        out.append((f"x_sb{r}_qdom", np.full((1, 1), float(k), np.float32), 1e9))
    x0, y0, x1, y1 = box
    crop = tctx.full[y0:y1, x0:x1]
    h, w = crop.shape
    ch, cw = min(h, 48), min(w, 48)
    t = np.ascontiguousarray(crop[(h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw])
    if t.std() >= 3 / 255 and min(t.shape) >= 4:
        gh, gw = tctx.shape[0] // DOWN, tctx.shape[1] // DOWN
        m = cv2.resize(_ncc_map(tctx.full, t, max(3, min(t.shape) // 2) | 1), (gw, gh), interpolation=cv2.INTER_AREA)
        out += [("x_ncc1", m, DOWN), ("x_ncc1_b6", _blur(m, 6), DOWN)]
    else:
        out += [("x_ncc1", None, DOWN), ("x_ncc1_b6", None, DOWN)]
    return out


def tex_features_at(tctx, qc, ys, xs):
    cols, names = [], []
    n = len(ys)
    for k, v, st in qc["maps"]:
        cols.append(np.full(n, np.nan, np.float32) if v is None else sample_grid(v, st, ys, xs))
        names.append(k)
        if k in ("ncc_c", "ncc_best"):
            names.append(k + "_rank")
            cols.append(np.full(n, np.nan, np.float32) if v is None
                        else np.interp(cols[-1], qc["quant"][k], np.linspace(0, 1, 101)))
    for k, v in qc["consts"].items():
        cols.append(np.full(n, v, np.float32)); names.append(k)
    for k, v, st in qc.get("extra", []):
        cols.append(np.full(n, np.nan, np.float32) if v is None else sample_grid(v, st, ys, xs))
        names.append(k)
    return np.stack(cols, 1).astype(np.float32), names
