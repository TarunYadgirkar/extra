# Agent guide: hatch matching solution

This file is the starting point for any agent continuing work here. The upstream challenge docs are README.md, SPECIFICATION.md and BENCHMARK.md. The full results history, including the numbers for every experiment tried, is in `solution/EXPERIMENTS.md`. Read that file before proposing anything, because many obvious ideas have already been measured and failed.

## Task

The input is a construction drawing (grayscale PNG, about 7200x4800) plus a query box drawn around one hatch pattern. The output is a binary mask, at native resolution, of every region in the drawing that has the same pattern. Scoring is done by `evaluate.py`. It computes IoU only on the labeled "known" pixels, averages that per document, then averages across documents (document-macro). The upstream reference scored 73.7% on a different suite, and the upstream target is 75%+.

## Current state (2026-09-23)

The official val document-macro IoU is **0.9241**, with recall 0.981 and precision 0.710 over 57 queries in 13 documents. The code lives on branch `claude/dino-head`, pushed to remote `mine` (github.com/TarunYadgirkar/extra). The remote `origin` is upstream TruTec-AI. Inference takes about 22 s per query on an M5 Pro (MPS).

The pipeline is `solution/infer.py`:
1. `features.py`: frozen DINOv2 ViT-S/14 reg4 patch features (timm), tiled, at 0.5x and 1.0x scale.
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

## Rebuilding the head from scratch

`cache/` is gitignored. What it keeps is `cache/feat_s_{0.5,1.0}` (the DINOv2-S grids for every train and val image, about 12 GB) and `cache/domains` (label pixels per example). Everything else was deleted and has to be rebuilt:

```bash
python solution/cache_feats.py train s 0.5   # repeat for val, and for scale 1.0 (skips existing)
python solution/devdata.py                   # label domains (skips existing)
python solution/build_rows.py val v3 && python solution/build_rows.py train v3 && python solution/build_rows.py train v3 uniform
OMP_NUM_THREADS=4 python solution/exp/texture/build_tx.py val tx_v2 --extra
OMP_NUM_THREADS=4 python solution/exp/texture/build_tx.py train tx_v2 --extra
OMP_NUM_THREADS=4 python solution/exp/texture/cv_tx.py tx_v2 "s0_,s1_,qs,b6,b16,qb,ncc,q_" --save
cp "cache/head_tx_v2_s0_+s1_+qs+b6+b16+qb+ncc+q_.joblib" solution/weights/head.joblib
```

`solution/exp/texture/txfeat.py` and `solution/texfeat.py` contain the same code. If you change one, change the other, or better, make the exp copy import from `solution/`.

## Evaluation protocol (use it, don't invent a new one)

- **cv** comes from `solution/cv.py` (and `exp/texture/cv_tx.py` or `exp/head/hcv.py`, which also save out-of-fold predictions). It assigns all 126 training documents (62 real, 64 generated CAD) to four folds: sort document IDs, shuffle with `np.random.default_rng(0)`, assign `i % 4`. Each model excludes every real and CAD query in its held-out fold. Scoring includes only real queries, using up to 20,000 uniformly sampled known pixels per query and document-macro IoU. Real-document counts per fold are 22, 15, 7 and 18.
- **val**: 13 documents. It is noisy; one document moves the score by about 3 points.
- A change counts only if cv improves under a paired per-document bootstrap (`exp/backbone/cv_oof.py --compare`, `exp/head/hcv.py`) and val does not drop. Seed noise alone is about 0.003 cv.
- Confirm the final number end to end with `infer.py` + `evaluate.py`. `exp/calib/val_pp.py` reproduces the val score from cached grids without a GPU.

## What is known

- **Worked:** the texture columns (+0.06 cv, the largest gain) and blur plus hole-fill (+0.022 cv, +0.009 val).
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

- Fine-tuning the last DINO blocks with a query-conditioned objective (only frozen features were used so far).
- Label-noise-aware training, e.g. downweighting train queries whose positives are tone-only fills.
- Cheaper inference: `predict_proba` on about 2M grid points per query dominates the runtime. Skipping blank paper or using coarse-to-fine evaluation would help.
- Cross-query competition among queries on the same drawing gave +0.005 cv and +0.003 val. It was not adopted because it makes each output depend on the batch.

## Rules

- Labels, annotation rectangles and expected counts must never be inference inputs. Only the train split (real + generated_cad) may be used for training, and val is for selection only. Never fabricate or synthesize training data.
- Put experiments in `solution/exp/<name>/`, with cache files prefixed per experiment, and log the results as rows in `solution/EXPERIMENTS.md`. Keep that table contiguous, with no blank lines inside it.
- The user's Mac is shared. Run at most two heavy jobs at once, under `nice -n 15` with `OMP_NUM_THREADS=4`.
- Don't submit to TruTec (ryan@trutec.ai) without the user's go-ahead.

## Ongoing (2026-09-25)

Fine-tuning DINO (exp/ft) is the first open idea tried; see the ft1* rows and the last observation bullet in `solution/EXPERIMENTS.md`. Summary so far:
- Cross-fitted design (ft1): last 4 blocks fine-tuned per cv fold with a query-conditioned similarity loss; fold backbones produce the head's training rows, the all-train backbone produces the grids at inference. cv +0.012 (seed 1: +0.014, seed noise +0.0025), CI touching zero; official val 0.9420 / 0.9380 (seeds 0/1) vs 0.9241, with doc 796851 up 0.15 to 0.21 and e86e6e down 0.06. Queries under 40 px lose, everything larger gains.
- Strict nested cv (head on same-backbone rows) is +0.0007, so the cv gain depends on training the head on out-of-fold backbone tokens. The deployable analogue of that (ft1a) gets official val 0.9362.
- Fully nested cv (12 inner backbones, `cache/logs/ft1_nested_chain.sh`) reproduces the gain: 0.9030, +0.0132 [-0.0069,+0.0339], 21 docs better 9 worse. Nothing in flight.
- **Decision pending (Tarun's call):** three independent cv measurements agree at +0.013 and val is +0.012 to +0.018 on three runs, but no CI excludes zero because three documents lose 0.15 to 0.33 (mostly queries under 40 px). To adopt: copy `cache/ft_w/ft1_fall.pt` (about 30 MB, last 4 blocks + norm) to `solution/weights/dino_ft.pt`, load it in `features.load_model` the way `exp/ft/ft_common.load_finetuned` does, install `cache/head_ft1_tx_v2_s0_+s1_+qs+b6+b16+qb+ncc+q_.joblib` as `solution/weights/head.joblib`, then confirm 0.9420 end to end with `infer.py` + `evaluate.py` (val_cached_ft.py already reproduced it from cached grids). The rebuild recipe becomes: prep_ft.py, train_ft.py ft1 {0,1,2,3,all}, extract_ft.py ft1, build_rows_bb.py with GRIDS=feat_ft1_*, TXBASE=ft1 cv_tx.py --save.
- If not adopted, the next open idea is label-noise-aware training; a cheap first cut is to downweight train queries whose positives are tone-only fills (see the `exp/evidence` rows for how to find them).
- Not adopted, nothing changed in `solution/infer.py` or `solution/weights/`. Nothing has been submitted.
- Rebuilt caches this session: rows v3 / tx_v2 (baseline cv reproduces 0.8899 exactly), feat_ft1*, ft_gray, ft_lab, ft_w. `cache/logs/*.sh` are the chain scripts used.
