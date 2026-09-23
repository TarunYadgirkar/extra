"""Dev-only: doc-macro per-query AUC of single features on val (sanity check of separability)."""
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from devdata import examples, ROOT  # noqa: E402

tag = sys.argv[1]
d = np.load(ROOT / "cache" / f"rows_val_{tag}.npz")
X, Y, Q, names = d["X"], d["Y"], d["Q"], list(d["names"])
exs = examples("val")
res = {}
for j, n in enumerate(names):
    col = X[:, j]
    if np.nanstd(col) == 0:
        continue
    docs = {}
    for i in np.unique(Q):
        m = Q == i
        y, v = Y[m], col[m]
        if y.all() or not y.any() or np.isnan(v).all():
            continue
        v = np.nan_to_num(v, nan=np.nanmin(v))
        a = roc_auc_score(y, v)
        docs.setdefault(exs[i]["document_id"], []).append(max(a, 1 - a))
    res[n] = np.mean([np.mean(x) for x in docs.values()])
for n, a in sorted(res.items(), key=lambda t: -t[1])[:40]:
    print(f"{n:16s} {a:.3f}")
