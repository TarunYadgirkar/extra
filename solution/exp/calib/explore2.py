"""Hole filling / combos sweep."""
import sys
from explore1 import run, base, fill_holes, median, blur, boot, fmt  # noqa

def chain(*fs):
    def f(g, st):
        for h in fs:
            g = h(g, st)
        return g
    return f

cands = [("fill", fill_holes(a)) for a in (40000, 100000, 250000, 1000000, 4000000)]
cands += [("med42+fill100k", chain(median(42), fill_holes(100000))), ("blur14+fill100k", chain(blur(14), fill_holes(100000))),
          ("fill100k+med42", chain(fill_holes(100000), median(42)))]
for name, fn in cands:
    r = run(name, fn)
    print(f"{name:16s}", "train", round(r["train"][0], 4), fmt(boot(base["train"][1], r["train"][1])),
          "| val", round(r["val"][0], 4), fmt(boot(base["val"][1], r["val"][1])), flush=True)
