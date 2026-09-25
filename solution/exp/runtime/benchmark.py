import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'solution'))
import infer
from features import load_gray
from pixfeat import ImageContext
from texfeat import TexContext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((ROOT / 'validation-inputs.json').read_text())['examples']
    selected = list({r['image']: r for r in rows}.values())[:args.queries]
    head = joblib.load(ROOT / 'solution/weights/head.joblib')
    fast = infer.predict_probabilities
    results = []
    for index, row in enumerate(selected):
        gray = load_gray(ROOT / 'dataset' / row['image'])
        stem = Path(row['image']).stem[:16]
        ctx = ImageContext(gray, [(np.load(ROOT / f'cache/feat_s_{scale}/{stem}.npy'), 14 / scale)
                                  for scale in (0.5, 1.0)])
        tctx = TexContext(gray)
        timings, masks = {}, {}
        order = ('sklearn', 'binned') if index % 2 == 0 else ('binned', 'sklearn')
        for method in order:
            infer.predict_probabilities = fast if method == 'binned' else lambda h, x: h.predict_proba(x)[:, 1]
            start = time.perf_counter()
            masks[method] = infer.predict_mask(head, ctx, tctx, row['query_box'], 0.5)
            timings[method] = time.perf_counter() - start
        changed = int(np.count_nonzero(masks['sklearn'] != masks['binned']))
        result = {'id': row['id'], 'image': row['image'], 'seconds': timings, 'changed_pixels': changed}
        results.append(result)
        print(json.dumps(result), flush=True)
        if changed:
            raise RuntimeError('Binned inference changed the prediction')
    infer.predict_probabilities = fast
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'timing_scope': 'predict_mask only; cached DINO and image contexts excluded',
                                      'examples': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
