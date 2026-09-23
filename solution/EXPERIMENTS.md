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
| tx_v1 all | v3 + 50 query-conditioned texture cols (exp/texture/txfeat.py): log-polar power spectrum at 1x/0.5x vs query, ink-blob size-class densities vs query, NCC of query crops (+rot90/flip) at 0.5x | 0.893/0.890/0.885/0.877 | 0.914/0.917/0.919/0.917 | 0.9152 (tx_v1f head, solution/infer.py) |
| tx_v1 -ncc | v3 + spectrum + blob only | 0.883/0.883/0.880/0.873 | 0.910/0.910/0.909/0.905 | – |
| tx_v1 -spec | v3 + blob + NCC only | 0.877/0.874/0.870/0.862 | 0.914/0.917/0.916/0.913 | – |
| tx_v1 -blob | v3 + spectrum + NCC only | 0.846/0.844/0.839/0.830 | 0.883/0.887/0.886/0.878 | – |
| tx_v1 ncc | v3 + NCC only | 0.838/0.839/0.836/0.828 | 0.882/0.888/0.891/0.889 | – |
| tx_v2 all | tx_v1 + 17 x_ cols: region-pooled (box-blurred) ncc/spectrum/blob maps, component shape-class (size x elongation) densities, native-res NCC | 0.889/0.885/0.880/0.872 | 0.917/0.921/0.921/0.915 | – |
| tx_v2 v1+x_sb | tx_v1 cols + shape-class blob densities only | 0.885/0.885/0.881/0.871 | 0.922/0.927/0.926/0.924 | – |
| tx_v2 v1+x_ncc1 | tx_v1 cols + native-res NCC only | 0.884/0.881/0.875/0.867 | 0.908/0.912/0.914/0.915 | – |
| tx_v1 deep | tx_v1 all, max_leaf_nodes 127, 600 iters | 0.885/0.882/0.876/0.869 | 0.914/0.915/0.913/0.908 | – |
| tx_v1f | tx_v1 cols recomputed with current txfeat.py (tile/center alignment fix in q_nccself), groups `s0_,s1_,qs,b6,b16,qb,ncc,q_` of rows tx_v2; head cache/head_tx_v2_s0_+s1_+qs+b6+b16+qb+ncc+q_.joblib | 0.892/0.890/0.885/0.877 | 0.912/0.915/0.917/0.916 | 0.9152 (evaluate.py, full 4px grid via exp/texture/val_cached.py with cached DINO-S grids) |
| nn_a2 | Neural head alone (exp/neural): query-conditioned dilated conv net on the s14 grid. Inputs are DINOv2-S PCA-64 at s14 and s28, 4 s14 tone channels, the 10 fixed v3 similarity maps plus their ranks, and learned query cross-attention with FiLM. Loss is BCE on known cells only, balanced per query. 1500 steps of 8 crops at 128 cells | 0.764/0.764/0.764/0.764 | 0.646/0.645/0.644/0.643 | – |
| nn_a2 stack | tx_v1 all + nn_a2 OOF logit column | 0.879/0.877/0.872/0.862 | 0.915/0.915/0.914/0.908 | – |
| nn_c1 | Auto-context residual: the nn_a inputs plus the dense OOF tx_v1 GBM logit at s14 (and its rank). Output = tx_v1 pixel logit + upsampled s14 correction. Unbounded, 1500 steps | 0.888/0.890/0.891/0.892 | 0.897/0.898/0.899/0.901 | – |
| nn_c3 | nn_c1 with the correction bounded to 3*tanh(z/3), an L2 penalty of 0.02 on it, in_drop 0.2, 1000 steps | 0.896/0.894/0.891/0.888 | 0.914/0.916/0.917/0.918 | – |

Observations
- Errors are dominated by whole look-alike negative patches being accepted (fn≈0), not boundary quality.
- Hard docs: e86e6e54f74f7d06 (sparse dashed grid on white; wide hatch variants), 796851cb2de5c8b5 (small hatched-on-gray query vs flat gray lot).
- CAD training data helps (+0.02 val on v1).
- Texture features (exp/texture): paired per-doc bootstrap on cv OOF @0.5, tx_v1f vs v3 = +0.062 [+0.036,+0.092], better on 32 docs, worse on 4 of 62. Adding x_sb to tx_v1f = -0.005 [-0.012,+0.000] (val +0.012 is noise). Blob (ink component size-class density vs query) is the strongest family; spectrum and NCC each add ~+0.01 cv on top. Extra CPU cost at inference: ~2.5s per drawing (spectra + components) + ~1.5s per query (NCC) + ~2.5s per query sampling the 4px grid.
- Residual val errors after tx: doc 796851 (queries 45/3: thin sparse hatch on gray, FPs on flat gray lot even though ncc≈0 and spectral energy ≈0 there; 28/56: FP on the same hatch rotated 90°), doc c3c44da (21/46), e86e6e (19/10). Cv worst doc cfd973b4 is symbol matching (plant circles), not texture.
- Backbone swaps are noise-level. Paired per-doc bootstrap over the 62 cv docs (exp/backbone/cv_oof.py --compare), cv@0.5 delta and 95% CI: bb_sv3-v3 +0.006 [-0.008,+0.020]; bb_sb-v3 -0.009 [-0.021,+0.001]; bb_v3s-v3 -0.005 [-0.022,+0.014]; bb_txv3-tx_v1 -0.001 [-0.014,+0.010]. Extraction per 7200x4800 drawing (MPS): DINOv2-S 0.5x+1x 6s, ViT-B 18s, DINOv3-S 6s; the extra grid pair also adds about 2s per query in features.
- Neural spatial head (exp/neural) does not beat tx_v1. On its own, the DINO-only net overfits badly: train loss falls to about 0.01, logits reach about ±15, and the output barely changes with threshold. Stacked onto tx_v1 it moves cv@0.5 by -0.013 (95% CI -0.037 to +0.006), because the GBM over-trusts the overconfident column. The best auto-context residual (nn_c3) moves cv@0.5 by +0.003 (CI -0.007 to +0.013) and val by -0.0015, which is noise. Scaling the unbounded nn_c1 correction by a factor chosen on cv (0.1 to 0.2) gives cv +0.006 to +0.009 (CI at 0.1: +0.000 to +0.012) but val -0.001 to -0.004. The dense OOF tx_v1 maps (cache/nn_gbm) reproduce bb_oof_bb_tx exactly (max abs difference 3e-8). The nn_c* cv is mildly optimistic, because each training query's map came from a fold GBM that saw the held-out fold.
