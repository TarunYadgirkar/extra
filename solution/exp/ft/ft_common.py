"""Shared pieces for the DINO fine-tuning experiment (exp/ft)."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import timm

HERE = Path(__file__).resolve().parent
SOL = HERE.parents[1]
ROOT = SOL.parent
sys.path.insert(0, str(SOL))
from devdata import examples  # noqa: E402

GRAY_DIR = ROOT / "cache" / "ft_gray"
LAB_DIR = ROOT / "cache" / "ft_lab"
W_DIR = ROOT / "cache" / "ft_w"
ARCH = "vit_small_patch14_reg4_dinov2.lvd142m"
P = 14
TILE_P = 37  # 518 px, same as features.TILE
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def stem(e):
    return Path(e["image"]).stem[:16]


def fold_assign(folds=4):
    exs = examples("train")
    docs = sorted({e["document_id"] for e in exs})
    rng = np.random.default_rng(0); rng.shuffle(docs)
    fo = {d: i % folds for i, d in enumerate(docs)}
    return np.array([fo[e["document_id"]] for e in exs])


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Backbone(nn.Module):
    """DINOv2-S with the last `n_train` blocks (and the final norm) trainable; earlier blocks run without grad."""

    def __init__(self, n_train=4):
        super().__init__()
        self.m = timm.create_model(ARCH, pretrained=True, dynamic_img_size=True, num_classes=0)
        self.n_train = n_train
        for p in self.m.parameters():
            p.requires_grad_(False)
        for blk in self.m.blocks[len(self.m.blocks) - n_train:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for p in self.m.norm.parameters():
            p.requires_grad_(True)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def trainable_state(self):
        return {k: v.detach().cpu() for k, v in self.m.state_dict().items()
                if k.startswith("norm.") or any(k.startswith(f"blocks.{i}.") for i in range(len(self.m.blocks) - self.n_train, len(self.m.blocks)))}

    def forward(self, x):
        """x (B,1,H,W) in [0,1] -> patch tokens (B, N, C), prefix tokens dropped."""
        m = self.m
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        nb = len(m.blocks) - self.n_train
        with torch.no_grad():
            x = m.patch_embed(x)
            x = m._pos_embed(x)
            x = m.norm_pre(x)
            for blk in m.blocks[:nb]:
                x = blk(x)
        for blk in m.blocks[nb:]:
            x = blk(x)
        x = m.norm(x)
        return x[:, m.num_prefix_tokens:]


def load_finetuned(path):
    """Plain timm model (as features.load_model returns) with fine-tuned weights loaded on top."""
    m = timm.create_model(ARCH, pretrained=True, dynamic_img_size=True, num_classes=0)
    if path is not None:
        sd = torch.load(path, map_location="cpu")["state"]
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
    return m.eval().to(device())
