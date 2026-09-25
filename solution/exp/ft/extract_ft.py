"""Dev: DINO grids from fine-tuned backbones, layout-compatible with cache/feat_s_{scale}.

extract_ft.py TAG          train images with their held-out fold's backbone, val images with the all-train backbone -> cache/feat_TAG_{0.5,1.0}
extract_ft.py TAG --all    train and val images with the all-train backbone -> cache/feat_TAGa_{0.5,1.0}
extract_ft.py TAG --strict K   train images NOT in fold K with fold K's backbone -> cache/feat_TAGsK_{0.5,1.0}; fold-K images are
                               symlinked from cache/feat_TAG_* (held-out extraction) so row builders find every image
extract_ft.py TAG --nested K   fold-j images (j != K) with backbone TAGnK_fj (trained without folds K and j) -> cache/feat_TAGnK_*;
                               fold-K images symlinked from cache/feat_TAG_* (backbone K, held-out extraction)
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_common import ROOT, W_DIR, load_finetuned, fold_assign, stem  # noqa: E402
from devdata import examples  # noqa: E402
from features import load_gray, dense_features  # noqa: E402

SCALES = (0.5, 1.0)


def main():
    tag, allmode = sys.argv[1], "--all" in sys.argv
    strict = int(sys.argv[sys.argv.index("--strict") + 1]) if "--strict" in sys.argv else None
    nested = int(sys.argv[sys.argv.index("--nested") + 1]) if "--nested" in sys.argv else None
    out_tag = tag + ("a" if allmode else f"s{strict}" if strict is not None else f"n{nested}" if nested is not None else "")
    wtag = tag if nested is None else f"{tag}n{nested}"
    if nested is not None:
        strict = nested
    outs = {s: ROOT / "cache" / f"feat_{out_tag}_{s}" for s in SCALES}
    for o in outs.values():
        o.mkdir(parents=True, exist_ok=True)
    tr, qf = examples("train"), fold_assign()
    jobs = {}  # weight key -> {image}
    for e, f in zip(tr, qf):
        if strict is not None:
            if f == strict:
                for s in SCALES:
                    dst = outs[s] / f"{Path(e['image']).stem[:16]}.npy"
                    dst.exists() or dst.symlink_to(ROOT / "cache" / f"feat_{tag}_{s}" / dst.name)
            else:
                jobs.setdefault(str(f) if nested is not None else str(strict), set()).add(e["image"])
            continue
        jobs.setdefault("all" if allmode else str(f), set()).add(e["image"])
    if strict is None:
        for e in examples("val"):
            jobs.setdefault("all", set()).add(e["image"])
    t = time.time(); n = 0
    for key, imgs in sorted(jobs.items()):
        todo = [i for i in sorted(imgs) if not all((outs[s] / f"{Path(i).stem[:16]}.npy").exists() for s in SCALES)]
        if not todo:
            continue
        model = load_finetuned(W_DIR / f"{wtag}_f{key}.pt")
        for img in todo:
            gray = load_gray(ROOT / "dataset" / img)
            for s in SCALES:
                f, _ = dense_features(model, gray, s)
                np.save(outs[s] / f"{Path(img).stem[:16]}.npy", f)
            n += 1
            print(key, n, img[:30], f"{time.time()-t:.0f}s", flush=True)
        del model; torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    print("extract done", flush=True)


if __name__ == "__main__":
    main()
