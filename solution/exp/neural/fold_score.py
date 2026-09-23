"""Dev: quick standalone score of one CV fold's cached nn predictions vs the v3 / tx_v1 OOF on the same docs.

python fold_score.py TAG FOLD
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_nn as E  # noqa: E402

tag, k = sys.argv[1], int(sys.argv[2])
exs, qfold = E.cv_split()
ri = E.row_index("train", "uniform")
_, Yu, Qu, _ = E.load_rows("train_u", "v3")
sel = np.concatenate([np.full(len(i), qfold[q] == k) for q, i in ri])
cols = []
for q, idx in ri:
    if qfold[q] != k:
        continue
    e = exs[q]; d = E.domain(e)
    g = np.load(E.PRED / tag / f"{e['id']}.npy").astype(np.float32)
    cols.append(E.sample_grid(g, E.S, d["ys"][idx].astype(np.float32), d["xs"][idx].astype(np.float32)))
p = E.sig(np.concatenate(cols))
for t in (0.3, 0.5, 0.7, 0.9):
    print(f"{tag} fold {k} t={t}", round(E.doc_macro(p, Yu[sel], Qu[sel], exs, t), 4))
for f in ("bb_oof_v3.npy", "bb_oof_bb_tx.npy"):
    print(f, round(E.doc_macro(np.load(E.ROOT / "cache" / f)[sel], Yu[sel], Qu[sel], exs, 0.5), 4))
