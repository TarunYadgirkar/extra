"""Dev: train/evaluate the component verifier on cached tables (cache/vf_comp).

Accounting: dropping component c of query q changes TP -= pos_c, FP -= tot_c - pos_c, FN += pos_c (pixels attributed
to their nearest 4px cell, restricted to the accepted native mask). Exact re-scoring lives in exact.py.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from vbase import CACHE, queries, doc_macro, boot, fmt

REL = ["r_s0_bcrot", "r_s1_bcrot", "r_s0_l1", "r_b6_bcarea", "r_b16_bccnt", "r_d0_cos", "r_d1_cos", "r_d1_lda", "p_mean", "p_q50",
       "m_ncc_best", "m_lda1", "m_tk1_rank"]


def load(split):
    exs, qs = queries(split)
    Q = []
    for i, e, f in qs:
        z = np.load(CACHE / "vf_comp" / f"{e['id']}.npz")
        Q.append(dict(i=i, e=e, fold=f, F=z["F"], names=list(z["names"]), pos=z["pos"], tot=z["tot"], qcomp=int(z["qcomp"]),
                      base=z["base"], sec=float(z["sec"])))
    return exs, Q


def add_rel(F, names, is_q):
    """Within-query context: each feature minus its max over the query's components, and minus the query component's value."""
    ix = [names.index(k) for k in REL]
    V = F[:, ix]
    mx = np.nanmax(V, 0, keepdims=True) if len(V) else V
    qv = V[is_q.astype(bool)] if is_q.any() else np.full((1, len(ix)), np.nan)
    return np.concatenate([F, V - mx, V - qv], 1), names + [f"rmax_{k}" for k in REL] + [f"rq_{k}" for k in REL]


def table(Q):
    """Rows = labelled non-query components of queries with positives. Returns X, names, meta arrays."""
    X, meta = [], []
    names = None
    for qi, q in enumerate(Q):
        tp, fp, fn = q["base"]
        if tp + fn == 0 or len(q["F"]) == 0:
            continue
        nm = q["names"]
        F, names = add_rel(q["F"], nm, q["F"][:, nm.index("is_q")])
        U = tp + fp + fn
        for c in range(len(F)):
            if c + 1 == q["qcomp"] or q["tot"][c] == 0:
                continue
            p, t = q["pos"][c], q["tot"][c]
            gain = (tp - p) / (U - (t - p)) - tp / U
            X.append(F[c]); meta.append((qi, c, p / t, t, gain))
    return np.array(X, np.float32), names, np.array(meta, float)


def apply(Q, drop, exs):
    """drop: {(qi, c)}. Returns doc-macro and per-doc dict from accounting."""
    per = {}
    for qi, q in enumerate(Q):
        tp, fp, fn = q["base"].astype(float)
        if tp + fn == 0:
            continue
        for c in range(len(q["pos"])):
            if (qi, c) in drop:
                p, t = q["pos"][c], q["tot"][c]
                tp -= p; fp -= t - p; fn += p
        per[q["i"]] = tp / (tp + fp + fn)
    return doc_macro(per, exs)


def doc_w(Q, meta):
    docs = {}
    for q in Q:
        docs[q["e"]["document_id"]] = docs.get(q["e"]["document_id"], 0) + 1
    return np.array([1.0 / docs[Q[int(m[0])]["e"]["document_id"]] for m in meta])


def make_model(kind):
    if kind == "lr":
        return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=10,
                                              l2_regularization=1.0, random_state=0)
    raise ValueError(kind)


def fit(X, meta, w, kind, target):
    if target == "gain":  # cost-sensitive: keep=1 when dropping hurts, weight = |IoU change|
        y = (meta[:, 4] <= 0).astype(int); sw = np.abs(meta[:, 4]) * w + 1e-6
    elif target == "gaincap":  # as gain, but each |IoU change| capped at 0.1 so one document cannot dominate
        y = (meta[:, 4] <= 0).astype(int); sw = np.minimum(np.abs(meta[:, 4]), 0.1) * w + 1e-6
    else:  # purity: keep=1 when majority positive, weight = known px share of the query union
        y = (meta[:, 2] >= 0.5).astype(int); sw = w * np.minimum(meta[:, 3] / meta[:, 3].mean(), 10)
    m = make_model(kind)
    if kind == "lr":
        m.fit(X, y, logisticregression__sample_weight=sw)
    else:
        m.fit(X, y, sample_weight=sw)
    return m


CUR = {"C1": ["d_p_mean", "p_q50", "larea_q", "r_s0_bc", "r_s0_bcrot", "r_s1_bc", "r_b6_bcarea", "r_b16_bccnt", "r_d1_cos", "r_d1_lda",
               "r_t_paper4_ad", "dm_s0_rotd", "dm_ncc_best", "rq_r_s0_bcrot", "rmax_r_s0_bcrot"],
       "C2": ["d_p_mean", "larea_q", "r_s0_bcrot", "r_b6_bcarea", "r_d1_lda", "dm_s0_rotd", "dm_ncc_best"],
       "C3": ["dm_s0_rotd", "dm_s1_rotd", "r_s0_bc", "r_s1_bc", "rq_r_s0_bcrot", "larea_q"]}


def cols(names, fam):
    if fam in CUR:
        return [names.index(k) for k in CUR[fam]]
    pick = []
    for j, n in enumerate(names):
        g = ("P" if not n.startswith(("r_", "m_", "dm_", "rmax_", "rq_")) else "R" if n.startswith("r_")
             else "M" if n.startswith(("m_", "dm_")) else "X")
        if g in fam:
            pick.append(j)
    return pick
