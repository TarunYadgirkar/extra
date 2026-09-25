# Environment

Measured on: Apple M5 Pro (15-core), 24 GB unified memory, macOS 27.0; PyTorch uses the MPS backend. CUDA and CPU also work through `features.device()` (CPU is much slower for the DINOv2 extraction).

Interpreter: CPython 3.12.13 (any 3.12.x). Install from the repository root:

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt -r solution/requirements.txt
```

`solution/requirements.txt` pins every inference dependency (torch 2.14.0, timm 1.0.30, huggingface-hub 1.32.0, numpy 2.5.3, pillow 11.3.0, opencv-python-headless 5.0.0.93, scipy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0). The scikit-learn pin matters: `infer.py` uses the fitted binning and native tree evaluator of `HistGradientBoostingClassifier` for speed and its regression tests (`python -m unittest discover -s solution/tests`) compare it against the public `predict_proba`.

No system packages beyond a working Python and, on Linux, the usual libgomp for scikit-learn.

Network: the first run downloads the pretrained DINOv2 ViT-S/14 reg4 checkpoint from the Hugging Face Hub (`timm/vit_small_patch14_reg4_dinov2.lvd142m`, pinned to revision `c04b5193082a8d5b0c4856c7937384a48136c5de`, about 88 MB) into the standard Hub cache; set `HF_HOME` to relocate it. Nothing else is fetched. The fine-tuned block weights (`solution/weights/dino_ft.pt`) and the head (`solution/weights/head.joblib`) ship in the repository.

Inference command (label-free inputs from `make_inputs.py`):

```bash
python solution/infer.py --inputs validation-inputs.json --data-root dataset --output-dir predictions
```
