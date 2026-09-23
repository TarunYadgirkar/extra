"""Dev-only: training-set construction + head experiments on tx_v1f features (cache/hd_* from prep.py).

Usage: hcv.py NAME [--val] [--save]      config NAME from CONFIGS -> cache/hd_oof_NAME.npy (+ hd_pval_NAME.npy)
       hcv.py --compare A B               paired per-document bootstrap on cv OOF (A/B: config names or 'tx_v1f')
Same 4-fold document split, scoring rows and doc-macro formula as solution/cv.py.
"""
import sys, time, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from devdata import examples, domain, score, ROOT  # noqa: E402
from cv import THS, doc_macro  # noqa: E402

C = ROOT / "cache"
NAMES = [str(n) for n in np.load(C / "hd_names.npy")]
BASE_OOF = {"tx_v1f": C / "tx_oof_tx_v2_s0_+s1_+qs+b6+b16+qb+ncc+q_.npy"}
BASE_PVAL = {"tx_v1f": C / "tx_pval_tx_v2_s0_+s1_+qs+b6+b16+qb+ncc+q_.npy"}
HGB = dict(learning_rate=0.06, max_iter=400, max_leaf_nodes=63, min_samples_leaf=200,
           l2_regularization=1.0, max_features=0.7, random_state=0)

INC = ["sm", "sr", "tk", "lda", "ldar"]
MONO_INC = ([f"{k}{i}{s}" for k in INC for i in (0, 1) for s in ("", "_rm", "_rmin", "_rank")]
            + ["ncc_c", "ncc_c_rank", "ncc_best", "ncc_best_rank", "ncc_max", "ncc_med", "ncc_cblur"]
            + [f"s{i}_{k}" for i in (0, 1) for k in ("bc", "bcrot", "bcrad", "bcang")]
            + [f"b{r}_{k}" for r in (6, 16) for k in ("bccnt", "bcarea")]
            + [f"{k}{r}_cos" for k in ("ori", "int") for r in (4, 12)])
MONO_DEC = ([n for n in NAMES if n.endswith("_ad")] + [f"s{i}_{k}" for i in (0, 1) for k in ("l1", "l1rad", "rotd")]
            + [f"{k}{r}_l1" for k in ("ori", "int") for r in (4, 12)])
MONO_TEX = ["ncc_c", "ncc_c_rank", "ncc_best", "ncc_best_rank", "ncc_max", "ncc_med", "ncc_cblur"] + \
           [f"s{i}_{k}" for i in (0, 1) for k in ("bc", "bcrot", "bcrad", "bcang")] + \
           [f"b{r}_{k}" for r in (6, 16) for k in ("bccnt", "bcarea")]

D = dict(w="query", classbal=False, cad=1.0, u=0, u_w=0.0, u_bal=False, hard=None, params={}, seeds=(0,),
         bag=None, mono=None, drop=(), real_only=False, absd=False)
CONFIGS = {
    "base": {},
    "classbal": dict(classbal=True),
    "docw": dict(w="doc"),
    "docw_cb": dict(w="doc", classbal=True),
    "cad05": dict(cad=0.5),
    "cad2": dict(cad=2.0),
    "u5k": dict(u=5000, u_w=0.5),
    "u5k_bal": dict(u=5000, u_w=0.5, u_bal=True),
    "uall": dict(u=20000, u_w=0.5),
    "hard_is": dict(hard=dict(mine="insample", tau=0.3, h_w=0.5)),
    "hard_oof": dict(hard=dict(mine="oof", tau=0.3, h_w=0.5)),
    "hard_oof_kp": dict(hard=dict(mine="oof", tau=0.5, h_w=0.25, keep_prior=True)),
    "hard_is_kp": dict(hard=dict(mine="insample", tau=0.5, h_w=0.25, keep_prior=True)),
    "mono": dict(mono="all"),
    "mono_tex": dict(mono="tex"),
    "seed1": dict(seeds=(1,)),
    "absd": dict(absd="free"),
    "icst3": dict(icst="3"),
    "icst2": dict(icst="2"),
    "icst2t": dict(icst="2t"),
    "icst2t_s3": dict(icst="2t", seeds=(0, 1, 2)),
    "drop_qsize": dict(drop=("qw", "qh", "n0", "n1")),
    "drop_qconst": dict(drop=("q_", "qs", "qb", "selfm", "n0", "n1", "qw", "qh")),
    "drop_qtone": dict(drop=("q_mean", "q_std", "q_ink", "q_bg", "q_paper", "q_ori", "q_int", "selfm", "n0", "n1", "qw", "qh")),
    "absd_mono": dict(absd="mono"),
    "seeds5": dict(seeds=(0, 1, 2, 3, 4)),
    "bagdoc5": dict(seeds=(0, 1, 2, 3, 4), bag="doc"),
    "lr03_800": dict(params=dict(learning_rate=0.03, max_iter=800)),
    "l2_10": dict(params=dict(l2_regularization=10.0)),
    "msl1000": dict(params=dict(min_samples_leaf=1000)),
    "mf04": dict(params=dict(max_features=0.4)),
    "leaf31": dict(params=dict(max_leaf_nodes=31, max_iter=600)),
}


def cfg(name):
    base, *mods = name.split("+")
    c = {**D, **CONFIGS[base]}
    for m in mods:
        c.update({k: v for k, v in CONFIGS[m].items()})
    return c


def load(split, rows=None):
    X = np.load(C / f"hd_{split}_X.npy", mmap_mode="r")
    Y, Q = np.load(C / f"hd_{split}_Y.npy"), np.load(C / f"hd_{split}_Q.npy")
    if rows is not None:
        return np.ascontiguousarray(X[rows]), Y[rows], Q[rows]
    return X, Y, Q


def folds(exs, k=4):
    docs = sorted({e["document_id"] for e in exs})
    rng = np.random.default_rng(0); rng.shuffle(docs)
    fold_of = {d: i % k for i, d in enumerate(docs)}
    return np.array([fold_of[e["document_id"]] for e in exs])


def query_totals(c, qs, exs):
    """Total weight per training query id."""
    tot = {}
    by_doc = {}
    for q in qs:
        by_doc.setdefault(exs[q]["document_id"], []).append(q)
    per_doc = len(qs) / len(by_doc)
    for d, ql in by_doc.items():
        for q in ql:
            w = 1.0 if c["w"] == "query" else per_doc / len(ql)
            if exs[q]["kind"] != "real":
                w *= c["cad"]
            tot[q] = w
    return tot


def spread(Y, Q, tot, bal):
    """Row weights: each query's total spread over its rows (optionally half to each class)."""
    w = np.zeros(len(Y))
    for q in np.unique(Q):
        m = Q == q
        if bal and Y[m].any() and (~Y[m]).any():
            mp, mn = m & Y, m & ~Y
            w[mp] = tot[q] / 2 / mp.sum(); w[mn] = tot[q] / 2 / mn.sum()
        else:
            w[m] = tot[q] / m.sum()
    return w


def sel_u(Qu, qs, n, seed=0):
    rng = np.random.default_rng(seed + 1)
    out = []
    for q in qs:
        idx = np.flatnonzero(Qu == q)
        if len(idx):
            out.append(idx if len(idx) <= n else np.sort(rng.choice(idx, n, replace=False)))
    return np.concatenate(out) if out else np.zeros(0, int)


ABS = ["s0_ed", "s1_ed", "s0_peakd", "s1_peakd", "s0_centd", "s1_centd",
       "b6_lcnt", "b6_larea", "b6_ldom", "b16_lcnt", "b16_larea", "b16_ldom"]
ABS_I = [NAMES.index(n) for n in ABS]


def design(c, X, cols):
    Z = X[:, cols]
    if c["absd"]:
        Z = np.concatenate([Z, np.abs(X[:, ABS_I])], 1)
    return Z


def dnames(c, cols):
    return [NAMES[j] for j in cols] + (["abs_" + n for n in ABS] if c["absd"] else [])


def monocst(c, cols):
    names = dnames(c, cols)
    inc = {"all": MONO_INC, "tex": MONO_TEX, None: []}[c["mono"]]
    dec = MONO_DEC if c["mono"] == "all" else []
    if c["absd"] == "mono":
        dec = dec + ["abs_" + n for n in ABS]
    if not inc and not dec:
        return None
    return [1 if n in inc else -1 if n in dec else 0 for n in names]


QCONST = ("q_", "qs", "qb", "selfm", "n0", "n1", "qw", "qh")
TONE = ("mean", "std", "ink", "bg", "paper", "ori", "int")
TEX = ("s0_", "s1_", "b6", "b16", "ncc", "abs_")


def family(n):
    if n.startswith(QCONST):
        return "q"
    if n.startswith(TEX):
        return "tex"
    if n.startswith(TONE):
        return "tone"
    return "dino"


def icst(c, cols):
    """Interaction groups: query constants join every group; other families only interact within their block."""
    if not c.get("icst"):
        return None
    fam = [family(n) for n in dnames(c, cols)]
    blocks = {"3": [("tone",), ("dino",), ("tex",)], "2": [("tone", "dino"), ("tex",)],
              "2t": [("tone",), ("dino", "tex")]}[c["icst"]]
    return [[j for j, f in enumerate(fam) if f == "q" or f in b] for b in blocks]


def fit_one(c, X, Y, w, seed, cols):
    p = {**HGB, **c["params"], "random_state": seed}
    mc = monocst(c, cols)
    if mc is not None:
        p["monotonic_cst"] = mc
    ic = icst(c, cols)
    if ic is not None:
        p["interaction_cst"] = ic
    return HistGradientBoostingClassifier(**p).fit(design(c, X, cols), Y, sample_weight=w / w.mean())


class Avg:
    def __init__(self, ms, cols, c=None):
        self.ms, self.cols, self.c = ms, cols, c

    def predict_proba(self, X):
        Z = design(self.c, X, self.cols) if self.c else X[:, self.cols]
        return np.mean([m.predict_proba(Z) for m in self.ms], 0)


def train_set(c, qs, exs, Xb, Yb, Qb, bidx, Xu, Yu, Qu):
    """Assemble (X, Y, w) for the training queries qs from balanced rows + optional uniform rows."""
    tot = query_totals(c, qs, exs)
    wb = spread(Yb, Qb, tot, c["classbal"])
    parts = [(Xb, Yb, Qb, wb * (1 - c["u_w"]) if c["u"] else wb)]
    if c["u"]:
        ui = sel_u(Qu, qs, c["u"])
        Yui, Qui = Yu[ui], Qu[ui]
        wu = spread(Yui, Qui, tot, c["u_bal"]) * c["u_w"]
        ureal = {q for q in np.unique(Qui)}
        wfix = np.array([1.0 if q in ureal else 1 / (1 - c["u_w"]) for q in Qb])
        parts[0] = (Xb, Yb, Qb, parts[0][3] * wfix)
        parts.append((np.ascontiguousarray(Xu[ui]), Yui, Qui, wu))
    return parts


def cat(parts):
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts]))


def mine(c, parts, qs, exs, cols, pool):
    """Hard negatives from pool=(X,Y,Q): pool negatives a stage-1 model scores above tau."""
    h = c["hard"]
    X, Y, Q, w = cat(parts)
    PX, PY, PQ = pool
    if h["mine"] == "insample":
        p = fit_one(c, X, Y, w, 0, cols).predict_proba(design(c, PX, cols))[:, 1]
    else:
        f = folds([exs[i] for i in range(len(exs))], 2)
        p = np.zeros(len(PY))
        for k in range(2):
            tr, te = f[Q] != k, f[PQ] == k
            p[te] = fit_one(c, X[tr], Y[tr], w[tr], 0, cols).predict_proba(design(c, PX[te], cols))[:, 1]
    hn = (~PY) & (p > h["tau"])
    tot = query_totals(c, qs, exs)
    wh = np.zeros(hn.sum())
    Qh = PQ[hn]
    for q in np.unique(Qh):
        m = Qh == q
        wh[m] = tot[q] * h["h_w"] / m.sum()
    print(f"  mined {hn.sum()} hard negs in {len(np.unique(Qh))} queries", flush=True)
    if h.get("keep_prior"):
        parts = [(Xp, Yp, Qp, wp.copy()) for Xp, Yp, Qp, wp in parts]
        npos = {}
        for _, Yp, Qp, _ in parts:
            for q, n in zip(*np.unique(Qp[Yp], return_counts=True)):
                npos[q] = npos.get(q, 0) + n
        for _, Yp, Qp, wp in parts:
            for q in np.unique(Qh):
                m = (Qp == q) & Yp
                if npos.get(q):
                    wp[m] += tot[q] * h["h_w"] / npos[q]
    return parts + [(PX[hn], PY[hn], PQ[hn], wh)]


def fit_cfg(c, qs, exs, Xb, Yb, Qb, Xu, Yu, Qu):
    cols = [j for j, n in enumerate(NAMES) if not n.startswith(tuple(c["drop"]))] if c["drop"] else list(range(len(NAMES)))
    parts = train_set(c, qs, exs, Xb, Yb, Qb, None, Xu, Yu, Qu)
    if c["hard"]:
        ui = sel_u(Qu, qs, 20000)
        PX = np.concatenate([Xb, Xu[ui]]); PY = np.concatenate([Yb, Yu[ui]]); PQ = np.concatenate([Qb, Qu[ui]])
        parts = mine(c, parts, qs, exs, cols, (PX, PY, PQ))
        del PX
    X, Y, Q, w = cat(parts)
    del parts
    if c["real_only"]:
        k = np.array([exs[q]["kind"] == "real" for q in Q]); X, Y, Q, w = X[k], Y[k], Q[k], w[k]
    ms = []
    for s in c["seeds"]:
        if c["bag"] == "doc":
            docs = sorted({exs[q]["document_id"] for q in qs})
            rng = np.random.default_rng(100 + s)
            cnt = dict(zip(docs, np.bincount(rng.integers(0, len(docs), len(docs)), minlength=len(docs))))
            ww = w * np.array([cnt[exs[q]["document_id"]] for q in Q])
            keep = ww > 0
            ms.append(fit_one(c, X[keep], Y[keep], ww[keep], s, cols))
        else:
            ms.append(fit_one(c, X, Y, w, s, cols))
    return Avg(ms, cols, c)


def run(name, do_val, save, cv_folds=4):
    c = cfg(name)
    exs = examples("train")
    qf = folds(exs)
    Xb, Yb, Qb = load("train")
    Xu, Yu, Qu = load("train_u")
    prob = np.zeros(len(Yu), np.float32)
    t0 = time.time()
    res = {"name": name}
    if not cv_folds:
        prob = np.load(C / f"hd_oof_{name}.npy")
    for k in range(cv_folds):
        qs = [i for i in range(len(exs)) if qf[i] != k]
        rb = np.flatnonzero(qf[Qb] != k)
        m = fit_cfg(c, qs, exs, np.ascontiguousarray(Xb[rb]), Yb[rb], Qb[rb], Xu, Yu, Qu)
        te = np.flatnonzero(qf[Qu] == k)
        prob[te] = m.predict_proba(np.ascontiguousarray(Xu[te]))[:, 1]
        print(f"  fold {k} {time.time()-t0:.0f}s", flush=True)
    if cv_folds:
        np.save(C / f"hd_oof_{name}.npy", prob)
    cvs = np.round([doc_macro(prob, Yu, Qu, exs, t) for t in THS], 4)
    res["cv"] = cvs.tolist()
    print(name, "cv", cvs, flush=True)
    if do_val or save:
        m = fit_cfg(c, list(range(len(exs))), exs, np.ascontiguousarray(Xb), Yb, Qb, Xu, Yu, Qu)
        if save:
            import joblib
            joblib.dump(m, C / f"hd_head_{name}.joblib")
        if do_val:
            Xv, Yv, Qv = load("val")
            pv = np.concatenate([m.predict_proba(np.ascontiguousarray(Xv[i:i + 500_000]))[:, 1] for i in range(0, len(Yv), 500_000)])
            np.save(C / f"hd_pval_{name}.npy", pv.astype(np.float32))
            vexs = examples("val")
            vals = np.round([score([(e, domain(e), pv[Qv == i] > t) for i, e in enumerate(vexs)])["doc_macro"] for t in THS], 4)
            res["val"] = vals.tolist()
            print(name, "val", vals, flush=True)
    with open(Path(__file__).parent / "logs" / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


def per_doc(prob, Y, Q, exs, t):
    docs = {}
    for i in np.unique(Q):
        m = Q == i; p = prob[m] > t; y = Y[m]
        if y.any():
            docs.setdefault(exs[i]["document_id"], []).append((p & y).sum() / (p | y).sum())
    return {d: np.mean(v) for d, v in docs.items()}


def oof_path(n):
    return BASE_OOF.get(n, C / f"hd_oof_{n}.npy")


def compare(a, b, ths=(0.5,)):
    Yu, Qu = np.load(C / "hd_train_u_Y.npy"), np.load(C / "hd_train_u_Q.npy")
    exs = examples("train")
    pa, pb = np.load(oof_path(a)), np.load(oof_path(b))
    for t in ths:
        da, db = per_doc(pa, Yu, Qu, exs, t), per_doc(pb, Yu, Qu, exs, t)
        diff = np.array([db[d] - da[d] for d in da])
        boot = np.random.default_rng(0).choice(diff, (5000, len(diff))).mean(1)
        print(f"t={t} {b}-{a}: {diff.mean():+.4f} [{np.quantile(boot, .025):+.4f},{np.quantile(boot, .975):+.4f}]"
              f" better {int((diff > 0.005).sum())} worse {int((diff < -0.005).sum())} of {len(diff)}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3], tuple(float(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 else (0.5,))
    else:
        run(sys.argv[1], "--val" in sys.argv, "--save" in sys.argv, 0 if "--nocv" in sys.argv else 4)
        compare("tx_v1f", sys.argv[1], THS)
