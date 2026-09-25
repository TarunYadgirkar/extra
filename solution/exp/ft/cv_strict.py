"""Strict nested cv for fine-tuned backbones: fold k's head trains on rows extracted with fold k's backbone
(rows_train_{TAG}s{k}) and is scored on the held-out fold of rows_train_u_{TAG}. Usage: cv_strict.py TAG"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "texture"))
import cv as cvmod  # noqa: E402
from cv_tx import make_loader  # noqa: E402
import cv_tx  # noqa: E402
from devdata import examples, ROOT  # noqa: E402
from ft_common import fold_assign  # noqa: E402

G = "s0_,s1_,qs,b6,b16,qb,ncc,q_"
tag = sys.argv[1]
exs = examples("train"); kinds = [e["kind"] for e in exs]
qfold = fold_assign()
cv_tx.BASE = tag
Xu, Yu, Qu, _ = make_loader("tx_v2", G.split(","))("train_u", None)
prob = np.zeros(len(Yu), np.float32)
for k in range(4):
    cv_tx.BASE = f"{tag}s{k}"
    X, Y, Q, _ = make_loader("tx_v2", G.split(","))("train", None)
    tr = qfold[Q] != k
    m = cvmod.fit(X[tr], Y[tr], Q[tr], kinds)
    te = qfold[Qu] == k
    prob[te] = m.predict_proba(Xu[te])[:, 1]
    del X
    print("fold", k, "done", flush=True)
np.save(ROOT / "cache" / f"tx_oof_{tag}strict_tx_v2_{G.replace(',', '+')}.npy", prob)
print(tag, "strict cv", np.round([cvmod.doc_macro(prob, Yu, Qu, exs, t) for t in cvmod.THS], 4), flush=True)
