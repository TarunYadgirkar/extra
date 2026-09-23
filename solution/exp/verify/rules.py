"""Dev: single-feature drop rules (drop non-query comp when feature on the 'bad' side of a cut), nested over folds."""
import numpy as np
from fit import load, table, apply
from vbase import boot, fmt

exs, Q = load("train"); vexs, V = load("val")
X, names, meta = table(Q); Xv, _, metav = table(V)
rf = np.array([Q[int(m[0])]["fold"] for m in meta])
key = [(int(m[0]), int(m[1])) for m in meta]; keyv = [(int(m[0]), int(m[1])) for m in metav]
base, vbase = apply(Q, set(), exs), apply(V, set(), vexs)
FEATS = ["r_s0_bc", "r_s0_bcrot", "r_s1_bc", "r_s1_bcrot", "r_s0_l1", "r_b6_bcarea", "r_b16_bccnt", "r_b6_lcnt", "r_d0_cos", "r_d1_cos",
         "r_d1_lda", "r_t_paper4_ad", "r_t_mean4_ad", "r_t_ink12_ad", "p_mean", "d_p_mean", "p_q50", "m_ncc_best", "m_ncc_med", "m_tk1",
         "rmax_r_s0_bcrot", "rq_r_s0_bcrot", "rq_p_mean", "rmax_p_mean", "dm_s0_rotd", "larea_q", "dist_q"]


def drops(v, keys, cut, sign):
    return {k for k, x in zip(keys, v) if not np.isnan(x) and sign * x < sign * cut}


def sub(Qs, qsel, dset):
    return apply([Qs[i] for i in qsel], {(qsel.index(a), c) for a, c in dset if a in qsel}, exs)


def best_cut(v, sign, qsel, keys):
    cands = [np.inf * -sign] + list(np.unique(np.nanquantile(v, np.linspace(0.02, 0.5 if sign > 0 else 0.98, 13))))
    return max(cands, key=lambda c: sub(Q, qsel, drops(v, keys, c, sign))[0])


for f in FEATS:
    j = names.index(f)
    for sign in (1, -1):  # +1: drop when low; -1: drop when high
        per = {}
        cuts = []
        for k in range(4):
            qin = [qi for qi, q in enumerate(Q) if q["fold"] != k]
            tr = rf != k
            c = best_cut(X[tr, j], sign, qin, [kk for kk, t in zip(key, tr) if t])
            cuts.append(c)
            qout = [qi for qi, q in enumerate(Q) if q["fold"] == k]
            per.update(sub(Q, qout, drops(X[rf == k, j], [kk for kk, t in zip(key, rf == k) if t], c, sign))[1])
        cfull = best_cut(X[:, j], sign, list(range(len(Q))), key)
        v = apply(V, drops(Xv[:, j], keyv, cfull, sign), vexs)
        b = boot(base[1], per)
        if abs(b[0]) > 1e-4 or b[3] + b[4] > 0:
            print(f"{f:18s} {'low' if sign > 0 else 'high'} nested cv {np.mean(list(per.values())):.4f} {fmt(b)} | cut {cfull:.3f} val {v[0]:.4f} {fmt(boot(vbase[1], v[1]))}", flush=True)
