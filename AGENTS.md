# Agent guide: hatch matching solution

This file is the starting point for any agent continuing work here. The upstream challenge docs are README.md, SPECIFICATION.md and BENCHMARK.md. The full results history, including the numbers for every experiment tried, is in `solution/EXPERIMENTS.md`. Read that file before proposing anything, because many obvious ideas have already been measured and failed.

## Task

The input is a construction drawing (grayscale PNG, about 7200x4800) plus a query box drawn around one hatch pattern. The output is a binary mask, at native resolution, of every region in the drawing that has the same pattern. Scoring is done by `evaluate.py`. It computes IoU only on the labeled "known" pixels, averages that per document, then averages across documents (document-macro). The upstream reference scored 73.7% on a different suite, and the upstream target is 75%+.

## Current state (2026-09-25)

The official val document-macro IoU is **0.9420**, with recall 0.970 and precision 0.799 over 57 queries in 13 documents (previous deployed version: 0.9241). The code lives on branch `claude/dino-head`, pushed to remote `mine` (github.com/TarunYadgirkar/extra). The remote `origin` is upstream TruTec-AI. Inference takes about 20 to 25 s per query on an M5 Pro (MPS), peak RSS 4.4 GB.

The pipeline is `solution/infer.py`:
1. `features.py`: DINOv2 ViT-S/14 reg4 patch features (timm, Hub revision pinned), tiled, at 0.5x and 1.0x scale. The last 4 transformer blocks and the final norm are fine-tuned (`weights/dino_ft.pt`, 28 MB; see `exp/ft`), the rest is the pretrained checkpoint.
2. `pixfeat.py`: per-location features on a 4 px grid. These are cosine, robust and top-k similarity to the query cells, exemplar-LDA (whitened against the image's own feature statistics), tone and ink maps, and orientation and intensity histograms, each compared against the query.
3. `texfeat.py`: query-conditioned texture columns. These are a log-polar power spectrum (line spacing x angle), ink-blob size-class densities, and normalized cross-correlation (NCC) of query crops.
4. A sklearn HistGradientBoosting head (`weights/head.joblib`, 162 columns: the 112 v3 columns followed by the texture groups `s0_,s1_,qs,b6,b16,qb,ncc,q_`).
5. Post-processing on the probability grid: a Gaussian blur with sigma 14 px, then every enclosed below-threshold hole not touching the border is filled, then the grid is thresholded at 0.5 and upsampled bilinearly.

## Setup

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt -r solution/requirements.txt
python download_data.py                     # dataset/ (gitignored, ~400 MB)
python make_inputs.py --manifest dataset/val.json --output validation-inputs.json
python solution/infer.py --inputs validation-inputs.json --data-root dataset --output-dir predictions
python evaluate.py --manifest dataset/val.json --data-root dataset --predictions predictions --output metrics.json
```

## Rebuilding the weights from scratch

`cache/` is gitignored. `cache/feat_s_{0.5,1.0}` holds the frozen DINOv2-S grids (about 12 GB, used by the baseline rows and by `cache_feats.py`), `cache/domains` the label pixels per example. The deployed head is trained on rows extracted with per-fold fine-tuned backbones (cross-fitting), and inference uses the backbone fine-tuned on all of train:

```bash
python solution/devdata.py                                   # label domains (skips existing)
python solution/cache_feats.py train s 0.5                   # frozen grids, repeat for val and scale 1.0 (only needed for the baseline rows)
python solution/build_rows.py val v3 && python solution/build_rows.py train v3 && python solution/build_rows.py train v3 uniform
OMP_NUM_THREADS=4 python solution/exp/texture/build_tx.py val tx_v2 --extra
OMP_NUM_THREADS=4 python solution/exp/texture/build_tx.py train tx_v2 --extra
python solution/exp/ft/prep_ft.py                            # raw gray images + cell label grids for the fine-tune
for f in 0 1 2 3 all; do python solution/exp/ft/train_ft.py ft1 $f steps=800 lr=1e-5; done   # about 12 to 20 min each on MPS
python solution/exp/ft/extract_ft.py ft1                     # train images by their held-out fold's backbone, val by the all-train one
export GRIDS=feat_ft1_0.5:28,feat_ft1_1.0:14
python solution/exp/backbone/build_rows_bb.py val ft1 && python solution/exp/backbone/build_rows_bb.py train ft1 && python solution/exp/backbone/build_rows_bb.py train ft1 uniform
TXBASE=ft1 OMP_NUM_THREADS=4 python solution/exp/texture/cv_tx.py tx_v2 "s0_,s1_,qs,b6,b16,qb,ncc,q_" --save
cp "cache/head_ft1_tx_v2_s0_+s1_+qs+b6+b16+qb+ncc+q_.joblib" solution/weights/head.joblib
cp cache/ft_w/ft1_fall.pt solution/weights/dino_ft.pt
```

The v3 rows are only needed for the baseline comparison (`exp/ft/compare.py`). `cache/logs/*.sh` in the working tree are the exact chains that produced the deployed weights.

`solution/exp/texture/txfeat.py` and `solution/texfeat.py` contain the same code. If you change one, change the other, or better, make the exp copy import from `solution/`.

## Evaluation protocol (use it, don't invent a new one)

- **cv** comes from `solution/cv.py` (and `exp/texture/cv_tx.py` or `exp/head/hcv.py`, which also save out-of-fold predictions). It assigns all 126 training documents (62 real, 64 generated CAD) to four folds: sort document IDs, shuffle with `np.random.default_rng(0)`, assign `i % 4`. Each model excludes every real and CAD query in its held-out fold. Scoring includes only real queries, using up to 20,000 uniformly sampled known pixels per query and document-macro IoU. Real-document counts per fold are 22, 15, 7 and 18.
- **val**: 13 documents. It is noisy; one document moves the score by about 3 points.
- A change counts only if cv improves under a paired per-document bootstrap (`exp/backbone/cv_oof.py --compare`, `exp/head/hcv.py`) and val does not drop. Seed noise alone is about 0.003 cv.
- Confirm the final number end to end with `infer.py` + `evaluate.py`. `exp/calib/val_pp.py` reproduces the val score from cached grids without a GPU.

## What is known

- **Worked:** the texture columns (+0.06 cv, the largest gain), blur plus hole-fill (+0.022 cv, +0.009 val), and fine-tuning the last DINO blocks with cross-fitted rows (+0.013 cv on three independent measurements, CI touching zero; +0.018 official val).
- **Did not work (measured, see EXPERIMENTS.md):**
  - ViT-B, DINOv3-S, or combined backbones
  - a neural spatial head, or stacking one on top
  - hard-negative mining, per-document or class reweighting, changing the CAD weight
  - monotonic or interaction constraints
  - bigger trees; seed or model blends (+0.0065 cv but noise on val, at 1–2 min per query)
  - per-query thresholds (the oracle gains +0.05, but the best threshold is bimodal and couldn't be predicted)
  - a component-level keep/drop verifier (AUC at chance)
  - evidence and orientation-strict features
- **The ceiling is set by label conflict.** Train labels a blank patch whose tone matches a textured query as positive about 92% of the time. The flat gray pavement fills in the training set are labeled as whole regions, yet val doc 796851 labels the same situation negative. Train labels are also not strict about 90° rotations, while val queries 28/56 expect them to be. Features that separate these cases already exist (`exp/evidence`), but they lower cv. Don't tune toward the val reading without train-side evidence.

## Open ideas not yet tried

- Fine-tuning variants: the adopted ft1 hurts queries under 40 px (one or two cells), so a size-aware fallback or a prototype built from a dilated query box might recover them; more blocks, a longer schedule, or a second seed averaged at the grid level were not tried.
- Label-noise-aware training, e.g. downweighting train queries whose positives are tone-only fills.
- Cheaper inference: `predict_proba` on about 2M grid points per query dominates the runtime. Skipping blank paper or using coarse-to-fine evaluation would help.
- Cross-query competition among queries on the same drawing gave +0.005 cv and +0.003 val. It was not adopted because it makes each output depend on the batch.

## Rules

- Labels, annotation rectangles and expected counts must never be inference inputs. Only the train split (real + generated_cad) may be used for training, and val is for selection only. Never fabricate or synthesize training data.
- Put experiments in `solution/exp/<name>/`, with cache files prefixed per experiment, and log the results as rows in `solution/EXPERIMENTS.md`. Keep that table contiguous, with no blank lines inside it.
- The user's Mac is shared. Run at most two heavy jobs at once, under `nice -n 15` with `OMP_NUM_THREADS=4`.
- Don't submit to TruTec (ryan@trutec.ai) without the user's go-ahead.

## Ongoing (2026-09-25)

The DINO fine-tune (exp/ft, tag ft1) is **adopted**: `solution/weights/dino_ft.pt` and the ft1 head are deployed, `features.load_model` loads them, official val is 0.9420 end to end. The evidence and the caveat (no cv CI excludes zero because three documents lose 0.15 to 0.33, mostly tiny queries) are in the ft1* rows and the last observation bullet of `solution/EXPERIMENTS.md`. `solution/report/REPORT.md`, `solution/ENVIRONMENT.md`, `solution/submission/` and `submission.json` are the submission package. Nothing has been sent to TruTec yet; that is Tarun's call.

Nothing is in flight. Next ideas are in "Open ideas" above.
