"""dense_features generalized to any patch size / timm backbone."""
import sys
from pathlib import Path
import numpy as np
import torch
import timm
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from features import device, load_gray  # noqa: F401

MODELS = {
    "s": ("vit_small_patch14_reg4_dinov2.lvd142m", 14),
    "b": ("vit_base_patch14_reg4_dinov2.lvd142m", 14),
    "v3s": ("vit_small_patch16_dinov3.lvd1689m", 16),
    "v3sp": ("vit_small_plus_patch16_dinov3.lvd1689m", 16),
    "v3b": ("vit_base_patch16_dinov3.lvd1689m", 16),
}


def load_model(name):
    arch, patch = MODELS[name]
    m = timm.create_model(arch, pretrained=True, dynamic_img_size=True, num_classes=0)
    m = m.eval().to(device())
    m.patch = patch
    return m


@torch.no_grad()
def dense_features(model, gray, scale=1.0, batch=4, tile_p=37, margin_p=4):
    """Same tiling scheme as features.dense_features; tile/margin in patches."""
    P = model.patch
    TILE, MARGIN = tile_p * P, margin_p * P
    img = Image.fromarray(gray)
    if scale != 1.0:
        img = img.resize((round(gray.shape[1] * scale), round(gray.shape[0] * scale)), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    H, W = a.shape
    gh, gw = -(-H // P), -(-W // P)
    Hp, Wp = gh * P, gw * P
    pad = np.ones((Hp + 2 * MARGIN + TILE, Wp + 2 * MARGIN + TILE), np.float32)
    pad[MARGIN:MARGIN + H, MARGIN:MARGIN + W] = a
    keep = TILE - 2 * MARGIN
    jobs = [(y, x) for y in range(0, Hp, keep) for x in range(0, Wp, keep)]
    C = model.num_features
    out = np.zeros((gh, gw, C), np.float16)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device()).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device()).view(1, 3, 1, 1)
    npre = model.num_prefix_tokens
    n, k = tile_p, keep // P
    for i in range(0, len(jobs), batch):
        chunk = jobs[i:i + batch]
        t = np.stack([pad[y:y + TILE, x:x + TILE] for y, x in chunk])
        x_t = torch.from_numpy(t).to(device()).unsqueeze(1).expand(-1, 3, -1, -1)
        tok = model.forward_features((x_t - mean) / std)[:, npre:]
        tok = tok.reshape(len(chunk), n, n, C).float().cpu().numpy()
        for (y, x), f in zip(chunk, tok):
            gy, gx = y // P, x // P
            hh, ww = min(k, gh - gy), min(k, gw - gx)
            out[gy:gy + hh, gx:gx + ww] = f[margin_p:margin_p + hh, margin_p:margin_p + ww]
    return out, P / scale
