"""Quick sweep of grid-level post-processing on s14 maps (train OOF + val)."""
import sys
import cv2
import numpy as np
from scipy import ndimage as ndi
from maps import Scorer, load_grid
from common import boot, fmt

src = sys.argv[1] if len(sys.argv) > 1 else "s14"
SPL = {s: Scorer(s) for s in ("train", "val")}
G = {s: {i: load_grid(src, e) for i, e, _ in S.qs} for s, S in SPL.items()}


def run(name, fn, t=0.5):
    out = {}
    for s, S in SPL.items():
        P = {i: S.probs(i, fn(g, st), st) for i, (g, st) in G[s].items()}
        out[s] = S.score(P, t)
    return out


def blur(sig_px):
    return lambda g, st: cv2.GaussianBlur(g, (0, 0), max(sig_px / st, 0.3))


def median(k_px):
    def f(g, st):
        k = max(3, int(round(k_px / st)) | 1)
        return ndi.median_filter(g, size=k)
    return f


def hyst(lo, hi):
    def f(g, st):
        lab, n = ndi.label(g > lo)
        if n == 0:
            return g
        mx = ndi.maximum(g, lab, np.arange(1, n + 1))
        keep = np.r_[False, mx > hi]
        return np.where(keep[lab], np.maximum(g, 0.51), np.minimum(g, 0.49))
    return f


def comp_mean(lo, hi):
    def f(g, st):
        lab, n = ndi.label(g > lo)
        if n == 0:
            return g
        mn = ndi.mean(g, lab, np.arange(1, n + 1))
        keep = np.r_[False, mn > hi]
        return np.where(keep[lab], np.maximum(g, 0.51), np.minimum(g, 0.49))
    return f


def min_area(a_px):
    def f(g, st):
        lab, n = ndi.label(g > 0.5)
        if n == 0:
            return g
        area = ndi.sum(np.ones_like(g), lab, np.arange(1, n + 1)) * st * st
        drop = np.r_[False, area < a_px]
        return np.where(drop[lab], 0.49, g)
    return f


def fill_holes(a_px):
    def f(g, st):
        m = g > 0.5
        lab, n = ndi.label(~m)
        if n == 0:
            return g
        area = ndi.sum(np.ones_like(g), lab, np.arange(1, n + 1)) * st * st
        fill = np.r_[False, area < a_px]
        return np.where(fill[lab], 0.51, g)
    return f


base = run("base", lambda g, st: g)
print("base", {s: round(v[0], 4) for s, v in base.items()}, flush=True)
if __name__ == "__main__":
    cands = [("blur", blur, (7, 14, 28, 56)), ("median", median, (42, 70, 126)),
             ("hyst.3>.8", lambda _: hyst(.3, .8), (0,)), ("hyst.5>.9", lambda _: hyst(.5, .9), (0,)),
             ("hyst.5>.95", lambda _: hyst(.5, .95), (0,)), ("hyst.4>.9", lambda _: hyst(.4, .9), (0,)),
             ("cmean.5>.7", lambda _: comp_mean(.5, .7), (0,)), ("cmean.5>.8", lambda _: comp_mean(.5, .8), (0,)),
             ("cmean.3>.6", lambda _: comp_mean(.3, .6), (0,)), ("minarea", min_area, (2000, 10000, 40000, 160000)),
             ("fillholes", fill_holes, (2000, 10000, 40000))]
    for name, mk, ps in cands:
        for p in ps:
            r = run(name, mk(p))
            print(f"{name:12s} {p:>7}", "train", round(r["train"][0], 4), fmt(boot(base["train"][1], r["train"][1])),
                  "| val", round(r["val"][0], 4), fmt(boot(base["val"][1], r["val"][1])), flush=True)
