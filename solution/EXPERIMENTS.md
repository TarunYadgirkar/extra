# Experiment log

Protocol: `solution/cv.py` reports two numbers per config at thresholds (0.4, 0.5, 0.6, 0.7).
- **cv**: 4-fold document-grouped CV on train real queries (train on other folds incl. CAD, score on a uniform 20k-pixel sample per held-out real query, doc-macro IoU).
- **val**: head fit on all train rows, scored on every known val pixel with the same formula as `evaluate.py`.
Select on cv first (62 docs), val second (13 docs, noisy). Default decision threshold is 0.5.
Official numbers come only from `solution/infer.py` + `evaluate.py`.

| Tag | Change | cv @0.4/0.5/0.6/0.7 | val @0.4/0.5/0.6/0.7 | official val |
|---|---|---|---|---|
| zero-shot | DINOv2-S cosine to query mean, val-tuned threshold | – | best 0.636 | – |
| v1 | HGB head: DINO-S sims at 0.5x+1x, tone/ink maps | – | 0.834/0.839/0.844/0.841 | 0.8415 |
| v2 | + orientation and intensity histograms vs query | 0.821/0.816/0.807/0.795 | 0.841/0.849/0.857/0.858 | – |
| v2 deep | max_leaf_nodes 127 | – | –/0.862/0.866/– | – |
| v3 | + exemplar-LDA (image-whitened) similarity | 0.829/0.828/0.822/0.814 | 0.860/0.868/0.871/0.869 | pending |
| v3 deep | max_leaf_nodes 127, 600 iters | 0.828/0.825/0.818/0.810 | 0.866/0.869/0.867/0.859 | – |

Observations
- Errors are dominated by whole look-alike negative patches being accepted (fn≈0), not boundary quality.
- Hard docs: e86e6e54f74f7d06 (sparse dashed grid on white; wide hatch variants), 796851cb2de5c8b5 (small hatched-on-gray query vs flat gray lot).
- CAD training data helps (+0.02 val on v1).
