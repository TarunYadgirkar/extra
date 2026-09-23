"""Query-conditioned dilated conv head on the stride-14 grid."""
import torch
import torch.nn as nn
import torch.nn.functional as F

NSIM = 20
DILS = (1, 2, 4, 8, 16, 1, 2, 4)


class Block(nn.Module):
    def __init__(self, d, dil):
        super().__init__()
        self.conv = nn.Conv2d(d, d, 3, padding=dil, dilation=dil)
        self.norm = nn.GroupNorm(8, d)

    def forward(self, x):
        return x + F.gelu(self.norm(self.conv(x)))


class Net(nn.Module):
    def __init__(self, cin=132, d=64, qd=64, dils=DILS, use_fixed=True, in_drop=0.1, absf=True, nsim=NSIM, resid=False,
                 dmax=0.0):
        super().__init__()
        self.use_fixed, self.absf, self.resid, self.dmax = use_fixed, absf, resid, dmax
        self.in_drop = nn.Dropout(in_drop)
        self.stem = nn.Sequential(nn.Linear(cin, d), nn.GELU(), nn.Linear(d, d))
        self.kproj = nn.Linear(d, qd)
        self.qproj = nn.Linear(d, qd)
        self.temp = nn.Parameter(torch.tensor(5.0))
        self.film = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d))
        nf = (2 if absf else 1) * d + 3 + (nsim if use_fixed else 0)
        self.fuse = nn.Conv2d(nf, d, 1)
        self.blocks = nn.Sequential(*[Block(d, k) for k in dils])
        self.out = nn.Conv2d(d, 1, 1)
        if resid:
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, sims, qtok, qmask):
        """x (B,H,W,cin), sims (B,H,W,NSIM), qtok (B,N,cin), qmask (B,N) bool -> logits (B,H,W)."""
        e = self.stem(self.in_drop(x))
        qe = self.stem(qtok)
        m = qmask.unsqueeze(-1).float()
        qmean = (qe * m).sum(1) / m.sum(1).clamp(min=1)
        qmax = qe.masked_fill(~qmask.unsqueeze(-1), -1e4).max(1).values
        k = F.normalize(self.kproj(e), dim=-1)
        qj = F.normalize(self.qproj(qe), dim=-1)
        proto = F.normalize((qj * m).sum(1), dim=-1)
        s_proto = torch.einsum("bhwc,bc->bhw", k, proto)
        cs = torch.einsum("bhwc,bnc->bhwn", k, qj).masked_fill(~qmask[:, None, None, :], -2)
        s_max = cs.max(-1).values
        top = cs.topk(min(3, cs.shape[-1]), dim=-1).values
        valid = (top > -1.5).float()
        s_top = (top * valid).sum(-1) / valid.sum(-1).clamp(min=1)
        g, b = self.film(torch.cat([qmean, qmax], -1)).chunk(2, -1)
        ef = e * (1 + g[:, None, None]) + b[:, None, None]
        diff = (e - qmean[:, None, None]).abs()
        lsim = torch.stack([s_proto, s_max, s_top], -1) * self.temp
        feats = ([ef] if self.absf else []) + [diff, lsim] + ([sims] if self.use_fixed else [])
        h = self.fuse(torch.cat(feats, -1).permute(0, 3, 1, 2))
        z = self.out(self.blocks(h))[:, 0]
        if self.dmax:
            z = self.dmax * torch.tanh(z / self.dmax)
        self.last_delta = z
        return z + sims[..., NSIM] * 4 if self.resid else z
