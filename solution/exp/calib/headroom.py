"""Oracle per-query threshold headroom on tx_v1f OOF (train_u) and val pixel probs."""
import numpy as np
from common import CACHE, TXF, examples, rows_yq, iou, doc_macro, boot, fmt

TS = np.round(np.arange(0.05, 0.96, 0.025), 3)

for split, f in (("train_u", f"tx_oof_{TXF}.npy"), ("val", f"tx_pval_{TXF}.npy")):
    exs = examples("train" if split == "train_u" else "val")
    Y, Q = rows_yq(split)
    P = np.load(CACHE / f)
    per = {}
    for i in np.unique(Q):
        m = Q == i
        per[i] = np.array([iou(P[m] > t, Y[m]) for t in TS])
    res = {}
    for t in (0.4, 0.5, 0.6):
        k = int(np.argmin(abs(TS - t)))
        res[t] = doc_macro({i: v[k] for i, v in per.items()}, exs)
    orc = doc_macro({i: np.nanmax(v) if not np.isnan(v).all() else np.nan for i, v in per.items()}, exs)
    print(split, {t: round(r[0], 4) for t, r in res.items()}, "oracle", round(orc[0], 4), "delta vs 0.5", fmt(boot(res[0.5][1], orc[1])))
    best_t = {i: TS[np.nanargmax(v)] for i, v in per.items() if not np.isnan(v).all()}
    bt = np.array(list(best_t.values()))
    print("  oracle t quantiles", np.percentile(bt, [10, 25, 50, 75, 90]).round(3))
    gain = sorted(((np.nanmax(v) - v[int(np.argmin(abs(TS - .5)))], i) for i, v in per.items() if not np.isnan(v).all()), reverse=True)[:8]
    print("  top gains", [(round(g, 3), exs[i]["id"][-6:], exs[i]["document_id"][:6], round(best_t[i], 2)) for g, i in gain])
