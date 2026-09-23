"""Label-free evidence features (prefix ev_): texture presence vs the query and orientation-strict comparisons.

Raw maps, each compared against the query's own value:
  ink:  adaptive-ink density log ratio at several radii (+ max-pooled presence)
  ge:   gradient energy (structure-tensor trace) log ratio
  st:   structure-tensor dominant orientation vs the query's (signed, coherence weighted)
  se:   max-pooled local spectral log energy minus the query's
  nr:   NCC of an inscribed query crop at 0/45/90/135/180 deg rotations (orientation gaps)
"""
import sys
from pathlib import Path
import cv2
import numpy as np

SOL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOL))
from pixfeat import sample_grid, _cells  # noqa: E402
from texfeat import spec_grid, _ncc_map, NCC_SCALE  # noqa: E402

DOWN = 4
INK_R = (4, 8, 16)
ST_R = (4, 12)
SE_POOL = 5  # cells of the stride-16 spectrum grid
ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)
EPS_INK, EPS_GE = 1e-3, 1e-5


def _small(a, gh, gw):
    return cv2.resize(a, (gw, gh), interpolation=cv2.INTER_AREA)


def _blur(m, r):
    return cv2.blur(m, (2 * r + 1, 2 * r + 1))


class EvContext:
    def __init__(self, gray, half=None):
        self.shape = gray.shape
        H, W = gray.shape
        gh, gw = H // DOWN, W // DOWN
        bg = cv2.dilate(gray, np.ones((9, 9), np.uint8))
        ink = ((bg.astype(np.int16) - gray) > 40).astype(np.float32)
        self.ink0 = _small(ink, gh, gw)
        self.ink = {r: _blur(self.ink0, r) for r in INK_R}
        self.inkmx = cv2.dilate(self.ink[4], np.ones((17, 17), np.uint8))
        g = gray.astype(np.float32) / 255.0
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        self.J0 = np.stack([_small(gx * gx, gh, gw), _small(gx * gy, gh, gw), _small(gy * gy, gh, gw)], -1)
        self.J = {r: _blur(self.J0, r) for r in ST_R}
        (_, loge) = spec_grid(gray, 1.0, 64, 16)
        self.loge = loge
        self.logemx = cv2.dilate(loge, np.ones((SE_POOL, SE_POOL), np.uint8))
        self.half = half if half is not None else cv2.resize(g, (round(W * NCC_SCALE), round(H * NCC_SCALE)),
                                                             interpolation=cv2.INTER_AREA)


def _tensor(J):
    a, b, c = J[..., 0], J[..., 1], J[..., 2]
    tr = a + c
    phi = np.arctan2(2 * b, a - c)
    coh = np.sqrt((a - c) ** 2 + 4 * b * b) / (tr + 1e-6)
    return tr, phi, coh


def _box_mean(grid, box, stride):
    iy, ix = _cells(box, stride, *grid.shape[:2])
    return grid[np.ix_(iy, ix)].reshape(-1, *grid.shape[2:]).mean(0)


def _rot_templates(half, box):
    """Inscribed square crops of the query at each angle (same size for every angle)."""
    x0, y0, x1, y1 = [int(round(v * NCC_SCALE)) for v in box]
    crop = half[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
    h, w = crop.shape
    S = min(h, w, 68)
    src = crop[(h - S) // 2:(h - S) // 2 + S, (w - S) // 2:(w - S) // 2 + S]
    s = int(S / np.sqrt(2))
    o = (S - s) // 2
    out = []
    for a in ANGLES:
        M = cv2.getRotationMatrix2D(((S - 1) / 2, (S - 1) / 2), a, 1.0)
        r = cv2.warpAffine(src, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out.append(np.ascontiguousarray(r[o:o + s, o:o + s]))
    return out


def ncc_rot_maps(ectx, box):
    gh, gw = ectx.shape[0] // DOWN, ectx.shape[1] // DOWN
    ts = _rot_templates(ectx.half, box)
    t0 = ts[0]
    if t0.std() < 3 / 255 or min(t0.shape) < 4:
        return None
    pool = max(3, min(t0.shape) // 2) | 1
    ms = [_small(_ncc_map(ectx.half, t, pool), gh, gw) for t in ts]
    return dict(zip(ANGLES, ms))


def ev_query(ectx, box):
    maps, consts = [], {}
    qi = float(_box_mean(ectx.ink0, box, DOWN))
    for r in INK_R:
        maps.append((f"ev_ink{r}", np.log((ectx.ink[r] + EPS_INK) / (qi + EPS_INK)), DOWN))
    maps.append(("ev_inkmx", np.log((ectx.inkmx + EPS_INK) / (qi + EPS_INK)), DOWN))
    consts["ev_q_ink"] = qi
    qtr, qphi, qcoh = _tensor(_box_mean(ectx.J0, box, DOWN))
    consts["ev_q_coh"] = float(qcoh)
    for r in ST_R:
        tr, phi, coh = _tensor(ectx.J[r])
        d = phi - qphi
        maps += [(f"ev_ge{r}", np.log((tr + EPS_GE) / (qtr + EPS_GE)), DOWN),
                 (f"ev_coh{r}", coh, DOWN),
                 (f"ev_stcos{r}", np.cos(d), DOWN),
                 (f"ev_stsin{r}", np.sin(d), DOWN),
                 (f"ev_stw{r}", np.cos(d) * np.sqrt(coh * qcoh), DOWN)]
    qe = float(_box_mean(ectx.loge, box, 16))
    maps.append(("ev_sepool", ectx.logemx - qe, 16))
    nm = ncc_rot_maps(ectx, box)
    names = ["ev_nr0", "ev_nr90", "ev_nr45", "ev_nr180", "ev_nrgap90", "ev_nrgap45", "ev_nrgap180", "ev_nrang"]
    if nm is None:
        maps += [(k, None, DOWN) for k in names]
        consts.update({"ev_q_sym90": np.nan, "ev_q_sym45": np.nan, "ev_q_sym180": np.nan})
    else:
        r90 = np.maximum(nm[90], nm[270])
        r45 = np.maximum.reduce([nm[45], nm[135], nm[225], nm[315]])
        r180 = nm[180]
        st = np.stack([nm[a] for a in ANGLES])
        k = st.argmax(0)
        ang = np.minimum(k, len(ANGLES) - k).astype(np.float32) * 45
        maps += [("ev_nr0", nm[0], DOWN), ("ev_nr90", r90, DOWN), ("ev_nr45", r45, DOWN), ("ev_nr180", r180, DOWN),
                 ("ev_nrgap90", nm[0] - r90, DOWN), ("ev_nrgap45", nm[0] - r45, DOWN),
                 ("ev_nrgap180", nm[0] - r180, DOWN), ("ev_nrang", ang, DOWN)]
        consts["ev_q_sym90"] = float(_box_mean(r90, box, DOWN))
        consts["ev_q_sym45"] = float(_box_mean(r45, box, DOWN))
        consts["ev_q_sym180"] = float(_box_mean(r180, box, DOWN))
        consts["ev_q_nr0"] = float(_box_mean(nm[0], box, DOWN))
    consts.setdefault("ev_q_nr0", np.nan)
    return {"maps": maps, "consts": consts}


def ev_features_at(qc, ys, xs):
    cols, names = [], []
    n = len(ys)
    for k, v, st in qc["maps"]:
        cols.append(np.full(n, np.nan, np.float32) if v is None else sample_grid(v, st, ys, xs))
        names.append(k)
    for k in sorted(qc["consts"]):
        cols.append(np.full(n, qc["consts"][k], np.float32)); names.append(k)
    return np.stack(cols, 1).astype(np.float32), names
