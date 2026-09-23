"""Dev: document-grouped CV (outer 4 folds = cv.py folds, nested threshold choice) + val for verifier configs.
python run.py [config ...]   config = FAM:KIND:TARGET, e.g. PR:hgb:gain"""
import sys
import numpy as np
from fit import load, table, apply, doc_w, fit, cols
from vbase import boot, fmt

TAUS = np.round(np.arange(0.1, 0.91, 0.1), 2)
exs, Q = load("train")
vexs, V = load("val")
X, names, meta = table(Q)
Xv, _, metav = table(V)
w, wv = doc_w(Q, meta), doc_w(V, metav)
rf = np.array([Q[int(m[0])]["fold"] for m in meta])
key = [(int(m[0]), int(m[1])) for m in meta]
keyv = [(int(m[0]), int(m[1])) for m in metav]
base = apply(Q, set(), exs)
vbase = apply(V, set(), vexs)
print(f"train rows {len(X)} (drop-worthy {(meta[:, 4] > 0).sum()}), val rows {len(Xv)} ({(metav[:, 4] > 0).sum()}); base cv {base[0]:.4f} val {vbase[0]:.4f}")


def score_drop(pk, keys, tau, QQ, ee, qsel=None):
    drop = {k for k, p in zip(keys, pk) if p < tau}
    if qsel is None:
        return apply(QQ, drop, ee)
    return apply([QQ[i] for i in qsel], {(qsel.index(a), c) for a, c in drop if a in qsel}, ee)


def run(cfg):
    fam, kind, target = cfg
    J = cols(names, fam)
    pk = np.full(len(X), np.nan)
    for k in range(4):
        tr = rf != k
        m = fit(X[tr][:, J], meta[tr], w[tr], kind, target)
        pk[rf == k] = m.predict_proba(X[rf == k][:, J])[:, 1]
    res = {t: score_drop(pk, key, t, Q, exs) for t in TAUS}
    # nested tau: for outer fold k, inner OOF over the other three folds picks tau
    per_doc = {}
    picks = []
    for k in range(4):
        inner = rf != k
        pin = np.full(len(X), np.nan)
        for j in range(4):
            if j == k:
                continue
            tr = (rf != k) & (rf != j)
            m = fit(X[tr][:, J], meta[tr], w[tr], kind, target)
            pin[rf == j] = m.predict_proba(X[rf == j][:, J])[:, 1]
        qin = [qi for qi, q in enumerate(Q) if q["fold"] != k]
        best = max(TAUS, key=lambda t: apply([Q[i] for i in qin], {(qin.index(a), c) for (a, c), p in zip(key, pin) if a in qin and p < t}, exs)[0])
        picks.append(best)
        qout = [qi for qi, q in enumerate(Q) if q["fold"] == k]
        d = apply([Q[i] for i in qout], {(qout.index(a), c) for (a, c), p in zip(key, pk) if a in qout and p < best}, exs)[1]
        per_doc.update(d)
    nested = float(np.mean(list(per_doc.values())))
    tbest = max(TAUS, key=lambda t: res[t][0])
    m = fit(X[:, J], meta, w, kind, target)
    pv = m.predict_proba(Xv[:, J])[:, 1]
    vres = {t: score_drop(pv, keyv, t, V, vexs) for t in TAUS}
    name = ":".join(cfg)
    print(f"{name:18s} cv@" + " ".join(f"{t}:{res[t][0]:.4f}" for t in TAUS[::2]) + f" | nested(taus {picks}) {nested:.4f} {fmt(boot(base[1], per_doc))}")
    print(f"{'':18s} val@" + " ".join(f"{t}:{vres[t][0]:.4f}" for t in TAUS[::2]) + f" | val@0.5 {fmt(boot(vbase[1], vres[0.5][1]))} | cv-best tau {tbest} val {vres[tbest][0]:.4f} {fmt(boot(vbase[1], vres[tbest][1]))}", flush=True)
    return pk, pv, per_doc


if __name__ == "__main__":
    cfgs = [tuple(c.split(":")) for c in sys.argv[1:]] or [(f, k, t) for f in ("P", "PR", "PRX", "PRMX") for k in ("lr", "hgb") for t in ("gain", "pure")]
    for c in cfgs:
        run(c)
