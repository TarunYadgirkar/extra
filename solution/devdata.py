"""Dev-only: scored-pixel domains for fast metric computation. Never imported by inference."""
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluate import reviewed_domains  # noqa: E402

CACHE = ROOT / "cache" / "domains"


def examples(split):
    return json.load(open(ROOT / "dataset" / f"{split}.json"))["examples"]


def domain(e):
    p = CACHE / f"{e['id']}.npz"
    if p.exists():
        d = np.load(p)
        return {k: d[k] for k in d.files}
    pos, known, blank, _ = reviewed_domains(e, ROOT / "dataset")
    ys, xs = np.nonzero(known)
    d = {"ys": ys.astype(np.int32), "xs": xs.astype(np.int32), "pos": pos[ys, xs], "blank": blank[ys, xs]}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **d)
    return d


def score(rows):
    """rows: list of (example, domain, predicted bool at domain pixels). Mirrors evaluate.aggregate."""
    docs, qi, tp_s, fp_s, fn_s = {}, [], 0, 0, 0
    per = []
    for e, d, m in rows:
        pos = d["pos"]
        tp = int((m & pos).sum()); fp = int((m & ~pos).sum()); fn = int((~m & pos).sum())
        tp_s += tp; fp_s += fp; fn_s += fn
        if pos.any():
            iou = tp / (tp + fp + fn)
            qi.append(iou); docs.setdefault(e["document_id"], []).append(iou)
            per.append((e["id"], iou, tp, fp, fn))
    dm = float(np.mean([np.mean(v) for v in docs.values()]))
    return {"doc_macro": dm, "query_mean": float(np.mean(qi)),
            "precision": tp_s / max(tp_s + fp_s, 1), "recall": tp_s / max(tp_s + fn_s, 1), "per": per}


if __name__ == "__main__":
    for s in ("val", "train"):
        for e in examples(s):
            domain(e)
        print(s, "cached")
