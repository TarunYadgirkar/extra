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
| v3 | + exemplar-LDA (image-whitened) similarity | 0.829/0.828/0.822/0.814 | 0.860/0.868/0.871/0.869 | 0.8697 |
| v3 deep | max_leaf_nodes 127, 600 iters | 0.828/0.825/0.818/0.810 | 0.866/0.869/0.867/0.859 | – |
| bb_b | v3 features, DINOv2 ViT-B/14 reg4 instead of ViT-S (0.5x+1x) | 0.825/0.823/0.818/0.809 | 0.859/0.871/0.876/0.873 | – |
| bb_sb | v3 + ViT-B grids (4 DINO grids, 158 cols) | 0.821/0.819/0.815/0.808 | 0.872/0.876/0.876/0.875 | – |
| bb_v3s | v3 features, DINOv3 ViT-S/16 (timm lvd1689m) instead of DINOv2-S, 0.5x+1x (stride 32/16) | 0.829/0.823/0.814/0.801 | 0.866/0.873/0.874/0.866 | – |
| bb_sv3 | v3 + DINOv3-S grids (4 DINO grids, 158 cols) | 0.835/0.834/0.828/0.818 | 0.871/0.872/0.869/0.863 | – |
| bb_txv3 | tx_v1 all + DINOv3-S grids (208 cols) | 0.893/0.889/0.882/0.871 | 0.917/0.920/0.920/0.916 | – |
| tx_v1 all | v3 + 50 query-conditioned texture cols (exp/texture/txfeat.py): log-polar power spectrum at 1x/0.5x vs query, ink-blob size-class densities vs query, NCC of query crops (+rot90/flip) at 0.5x | 0.893/0.890/0.885/0.877 | 0.914/0.917/0.919/0.917 | – |
| tx_v1 -ncc | v3 + spectrum + blob only | 0.883/0.883/0.880/0.873 | 0.910/0.910/0.909/0.905 | – |
| tx_v1 -spec | v3 + blob + NCC only | 0.877/0.874/0.870/0.862 | 0.914/0.917/0.916/0.913 | – |
| tx_v1 -blob | v3 + spectrum + NCC only | 0.846/0.844/0.839/0.830 | 0.883/0.887/0.886/0.878 | – |
| tx_v1 ncc | v3 + NCC only | 0.838/0.839/0.836/0.828 | 0.882/0.888/0.891/0.889 | – |
| tx_v2 all | tx_v1 + 17 x_ cols: region-pooled (box-blurred) ncc/spectrum/blob maps, component shape-class (size x elongation) densities, native-res NCC | 0.889/0.885/0.880/0.872 | 0.917/0.921/0.921/0.915 | – |

Observations
- Errors are dominated by whole look-alike negative patches being accepted (fn≈0), not boundary quality.
- Hard docs: e86e6e54f74f7d06 (sparse dashed grid on white; wide hatch variants), 796851cb2de5c8b5 (small hatched-on-gray query vs flat gray lot).
- CAD training data helps (+0.02 val on v1).
- Backbone swaps are noise-level. Paired per-doc bootstrap over the 62 cv docs (exp/backbone/cv_oof.py --compare), cv@0.5 delta and 95% CI: bb_sv3-v3 +0.006 [-0.008,+0.020]; bb_sb-v3 -0.009 [-0.021,+0.001]; bb_v3s-v3 -0.005 [-0.022,+0.014]; bb_txv3-tx_v1 -0.001 [-0.014,+0.010]. Extraction per 7200x4800 drawing (MPS): DINOv2-S 0.5x+1x 6s, ViT-B 18s, DINOv3-S 6s; the extra grid pair also adds about 2s per query in features.
