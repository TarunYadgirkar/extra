from pathlib import Path
import numpy as np
import torch
import timm
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
PATCH = 14
TILE = 518
MARGIN = 56
MODEL_NAMES = {
    "s": "vit_small_patch14_reg4_dinov2.lvd142m",
    "b": "vit_base_patch14_reg4_dinov2.lvd142m",
}
# Pretrained checkpoint pinned to an immutable Hub revision for reproducibility.
HUB_REVISIONS = {"s": "c04b5193082a8d5b0c4856c7937384a48136c5de"}
# Last 4 blocks + final norm of DINOv2-S, fine-tuned on the train split with a query-conditioned objective (exp/ft, tag ft1).
FT_WEIGHTS = {"s": Path(__file__).parent / "weights" / "dino_ft.pt"}


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def load_model(size="s", finetuned=True):
    kw = {}
    if size in HUB_REVISIONS:
        kw["pretrained_cfg_overlay"] = {"hf_hub_id": f"timm/{MODEL_NAMES[size]}@{HUB_REVISIONS[size]}"}
    m = timm.create_model(MODEL_NAMES[size], pretrained=True, dynamic_img_size=True, num_classes=0, **kw)
    if finetuned and size in FT_WEIGHTS:
        state = torch.load(FT_WEIGHTS[size], map_location="cpu")["state"]
        res = m.load_state_dict(state, strict=False)
        assert not list(res.unexpected_keys), res.unexpected_keys
    return m.eval().to(device())


def load_gray(path):
    return np.array(Image.open(path).convert("L"))


@torch.no_grad()
def dense_features(model, gray, scale=1.0, batch=8):
    """Patch features for the whole image on a stride-14 grid of the (rescaled) image.

    Tiles overlap by MARGIN px so every kept token saw context on all sides.
    Returns (gh, gw, C) float16 and the effective pixel stride in native coordinates.
    """
    img = Image.fromarray(gray)
    if scale != 1.0:
        img = img.resize((round(gray.shape[1] * scale), round(gray.shape[0] * scale)), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    H, W = a.shape
    gh, gw = -(-H // PATCH), -(-W // PATCH)
    Hp, Wp = gh * PATCH, gw * PATCH
    pad = np.ones((Hp + 2 * MARGIN + TILE, Wp + 2 * MARGIN + TILE), np.float32)
    pad[MARGIN:MARGIN + H, MARGIN:MARGIN + W] = a
    keep = TILE - 2 * MARGIN
    ys = list(range(0, Hp, keep)); xs = list(range(0, Wp, keep))
    C = model.num_features
    out = np.zeros((gh, gw, C), np.float16)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device()).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device()).view(1, 3, 1, 1)
    jobs = [(y, x) for y in ys for x in xs]
    npre = model.num_prefix_tokens
    for i in range(0, len(jobs), batch):
        chunk = jobs[i:i + batch]
        t = np.stack([pad[y:y + TILE, x:x + TILE] for y, x in chunk])
        x_t = torch.from_numpy(t).to(device()).unsqueeze(1).expand(-1, 3, -1, -1)
        x_t = (x_t - mean) / std
        tok = model.forward_features(x_t)[:, npre:]
        n = TILE // PATCH
        tok = tok.reshape(len(chunk), n, n, C).float().cpu().numpy()
        m = MARGIN // PATCH
        for (y, x), f in zip(chunk, tok):
            gy, gx = y // PATCH, x // PATCH
            k = keep // PATCH
            hh, ww = min(k, gh - gy), min(k, gw - gx)
            out[gy:gy + hh, gx:gx + ww] = f[m:m + hh, m:m + ww]
    return out, PATCH / scale
