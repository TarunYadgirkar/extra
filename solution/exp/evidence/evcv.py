"""Dev-only: tx_v1f columns (cache/hd_*) + ev_ columns (cache/ev_*), same folds/params/weights as exp/head/hcv.py.

Usage: evcv.py NAME [--val] [--save]   -> cache/ev_oof_NAME.npy (+ ev_pval_NAME.npy, ev_head_NAME.joblib)
       evcv.py --compare A B [ths]     paired per-doc bootstrap on cv OOF (A/B: config names or tx_v1f)
"""
import sys, time, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from devdata import examples, domain, score, ROOT  # noqa: E402
from cv import THS  # noqa: E402

C = ROOT / "cache"
HN = [str(n) for n in np.load(C / "hd_names.npy")]
EN = [str(n) for n in np.load(C / "ev_names.npy")] if (C / "ev_names.npy").exists() else []
BASE_OOF = C / "tx_oof_tx_v2_s0_+s1_+qs+b6+b16+qb+ncc+q_.npy"
HGB = dict(learning_rate=0.06, max_iter=400, max_leaf_nodes=63, min_samples_leaf=200,
           l2_regularization=1.0, max_features=0.7, random_state=0)

PRES_RAW = ("ev_ink", "ev_ge", "ev_sepool", "ev_q_ink")
ORI_RAW = ("ev_coh", "ev_st", "ev_nr", "ev_q_coh", "ev_q_sym", "ev_q_nr0")
PRES_DER = ("ev_pres", "ev_min1", "ev_min2", "ev_tonepres4", "ev_tonepres12", "ev_bgpres4", "ev_nrel", "ev_nratio")
ORI_DER = ("ev_sgap0", "ev_sgap1", "ev_ngap", "ev_nr0rel")
TONE = ("mean", "std", "ink", "bg", "paper", "int4", "int12")

CONFIGS = {
    "base": dict(raw=(), der=()),
    "raw": dict(raw=PRES_RAW + ORI_RAW, der=()),
    "all": dict(raw=PRES_RAW + ORI_RAW, der=PRES_DER + ORI_DER),
    "pres": dict(raw=PRES_RAW, der=PRES_DER),
    "ori": dict(raw=ORI_RAW, der=ORI_DER),
    "all_gate": dict(raw=PRES_RAW + ORI_RAW, der=PRES_DER + ORI_DER, gate=0.25),
    "pres_gate": dict(raw=PRES_RAW, der=PRES_DER, gate=0.25),
}


def sig(x):
    return 1 / (1 + np.exp(-np.nan_to_num(x, nan=0.0)))


def derive(H, E):
    h = lambda n: H[:, HN.index(n)]
    e = lambda n: E[:, EN.index(n)]
    pres = np.minimum.reduce([sig(2 * (e("ev_ink8") + 1)), sig(e("ev_ge12") + 2), sig(e("ev_sepool") + 2)])
    d = {
        "ev_pres": pres,
        "ev_min1": np.minimum.reduce([pres, np.nan_to_num(h("ncc_best_rank"), nan=1.0), h("s0_bcrot")]),
        "ev_min2": np.minimum.reduce([np.nan_to_num(h("ncc_c_rank"), nan=1.0), h("s0_bc"), h("s1_bc")]),
        "ev_tonepres4": h("paper4_ad") + (1 - pres),
        "ev_tonepres12": h("paper12_ad") + (1 - pres),
        "ev_bgpres4": h("bg4_ad") + (1 - pres),
        "ev_nrel": h("ncc_max") - h("q_nccself"),
        "ev_nratio": h("ncc_max") / np.maximum(h("q_nccself"), 0.05),
        "ev_sgap0": h("s0_bc") - h("s0_bcrot"),
        "ev_sgap1": h("s1_bc") - h("s1_bcrot"),
        "ev_ngap": h("ncc_c") - h("ncc_rot"),
        "ev_nr0rel": e("ev_nr0") - e("ev_q_nr0"),
    }
    return d


def design(c, H, E):
    """H: hd rows, E: ev rows (same rows). Returns float32 matrix and names."""
    H = np.array(H, np.float32)
    names = list(HN)
    parts = [H]
    if c.get("raw"):
        ri = [j for j, n in enumerate(EN) if n.startswith(c["raw"])]
        parts.append(np.asarray(E[:, ri], np.float32)); names += [EN[j] for j in ri]
    if c.get("der") or c.get("gate"):
        E = np.asarray(E, np.float32)
        d = derive(H, E)
        for k in c.get("der", ()):
            parts.append(d[k][:, None].astype(np.float32)); names.append(k)
        if c.get("gate"):
            off = d["ev_pres"] < c["gate"]
            for j, n in enumerate(HN):
                if n.startswith(TONE):
                    H[off, j] = np.nan
    return np.concatenate(parts, 1), names


def load(split, rows=None):
    H = np.load(C / f"hd_{split}_X.npy", mmap_mode="r")
    E = np.load(C / f"ev_{split}_X.npy", mmap_mode="r") if EN else np.zeros((len(H), 0), np.float32)
    Y, Q = np.load(C / f"hd_{split}_Y.npy"), np.load(C / f"hd_{split}_Q.npy")
    if rows is not None:
        return H[rows], E[rows], Y[rows], Q[rows]
    return H, E, Y, Q


def folds(exs, k=4):
    docs = sorted({e["document_id"] for e in exs})
    rng = np.random.default_rng(0); rng.shuffle(docs)
    fold_of = {d: i % k for i, d in enumerate(docs)}
    return np.array([fold_of[e["document_id"]] for e in exs])


def fit(c, X, Q, Y, seed=0):
    w = 1.0 / np.bincount(Q)[Q]
    return HistGradientBoostingClassifier(**{**HGB, "random_state": seed}).fit(X, Y, sample_weight=w / w.mean())


def predict(m, c, H, E, chunk=500_000):
    return np.concatenate([m.predict_proba(design(c, H[i:i + chunk], E[i:i + chunk])[0])[:, 1]
                           for i in range(0, len(H), chunk)]).astype(np.float32)


def run(name, do_val, save):
    c = CONFIGS[name]
    exs = examples("train")
    qf = folds(exs)
    Hb, Eb, Yb, Qb = load("train")
    Hu, Eu, Yu, Qu = load("train_u")
    Xb, names = design(c, Hb, Eb)
    print(name, "cols", len(names), flush=True)
    prob = np.zeros(len(Yu), np.float32)
    t0 = time.time()
    for k in range(4):
        tr = qf[Qb] != k
        m = fit(c, Xb[tr], Qb[tr], Yb[tr])
        te = np.flatnonzero(qf[Qu] == k)
        prob[te] = predict(m, c, Hu[te], Eu[te])
        print(f"  fold {k} {time.time()-t0:.0f}s", flush=True)
    np.save(C / f"ev_oof_{name}.npy", prob)
    res = {"name": name, "cv": [round(float(doc_macro(prob, Yu, Qu, exs, t)), 4) for t in THS]}
    print(name, "cv", res["cv"], flush=True)
    if do_val or save:
        m = fit(c, Xb, Qb, Yb)
        del Xb
        if save:
            import joblib
            joblib.dump({"model": m, "config": name, "names": names}, C / f"ev_head_{name}.joblib")
        if do_val:
            Hv, Ev, Yv, Qv = load("val")
            pv = predict(m, c, Hv, Ev)
            np.save(C / f"ev_pval_{name}.npy", pv)
            vexs = examples("val")
            res["val"] = [round(score([(e, domain(e), pv[Qv == i] > t) for i, e in enumerate(vexs)])["doc_macro"], 4)
                          for t in THS]
            print(name, "val", res["val"], flush=True)
    with open(Path(__file__).parent / "logs" / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


def per_doc(prob, Y, Q, exs, t):
    docs = {}
    for i in np.unique(Q):
        m = Q == i; p = prob[m] > t; y = Y[m]
        if y.any():
            docs.setdefault(exs[i]["document_id"], []).append((p & y).sum() / (p | y).sum())
    return {d: np.mean(v) for d, v in docs.items()}


def doc_macro(prob, Y, Q, exs, t):
    return np.mean(list(per_doc(prob, Y, Q, exs, t).values()))


def oof_path(n):
    return BASE_OOF if n == "tx_v1f" else C / f"ev_oof_{n}.npy"


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
        run(sys.argv[1], "--val" in sys.argv, "--save" in sys.argv)
        compare("tx_v1f", sys.argv[1], THS)
