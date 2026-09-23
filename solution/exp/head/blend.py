"""Dev-only: score probability averages of saved OOF/val predictions. Usage: blend.py A,B[,C...] [A2,B2 ...]
Names: hcv config names or tx_v1f. Prints cv, paired bootstrap vs tx_v1f, val and per-doc val for doc 796851."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hcv  # noqa: E402
from hcv import C, THS, doc_macro, examples, domain, score  # noqa: E402

exs, vx = examples("train"), examples("val")
Yu, Qu = np.load(C / "hd_train_u_Y.npy"), np.load(C / "hd_train_u_Q.npy")
Qv = np.load(C / "hd_val_Q.npy")
vidx = [np.flatnonzero(Qv == i) for i in range(len(vx))]
vdom = [domain(e) for e in vx]
pval = lambda n: np.load(hcv.BASE_PVAL.get(n, C / f"hd_pval_{n}.npy"))
for combo in sys.argv[1:]:
    ns = combo.split(",")
    name = "blend_" + "_".join(ns)
    p = np.mean([np.load(hcv.oof_path(n)) for n in ns], 0).astype(np.float32)
    np.save(C / f"hd_oof_{name}.npy", p)
    print(combo, "cv", np.round([doc_macro(p, Yu, Qu, exs, t) for t in THS], 4))
    hcv.compare("tx_v1f", name)
    if all((hcv.BASE_PVAL.get(n) or C / f"hd_pval_{n}.npy").exists() for n in ns):
        pv = np.mean([pval(n) for n in ns], 0)
        vals = []
        for t in THS:
            r = score([(e, vdom[i], pv[vidx[i]] > t) for i, e in enumerate(vx)])
            vals.append(round(r["doc_macro"], 4))
        ious = [iou for (qid, iou, *_) in score([(e, vdom[i], pv[vidx[i]] > 0.5) for i, e in enumerate(vx)])["per"]
                if next(e for e in vx if e["id"] == qid)["document_id"].startswith("doc-796851")]
        print("   val", vals, "796851@0.5", round(float(np.mean(ious)), 3), flush=True)
