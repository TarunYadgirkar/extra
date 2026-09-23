"""cv.py protocol for given tags. Usage: run_cv.py TAG [TAG ...] [--deep]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cv import cv

CONFIGS = {"base": ({}, 400), "deep": (dict(max_leaf_nodes=127, min_samples_leaf=100), 600)}

if __name__ == "__main__":
    tags = [a for a in sys.argv[1:] if not a.startswith("--")]
    which = ["base", "deep"] if "--deep" in sys.argv else ["base"]
    for tag in tags:
        for name in which:
            c, v, _ = cv(tag, *CONFIGS[name])
            print(tag, name, "cv", c, "val", v, flush=True)
