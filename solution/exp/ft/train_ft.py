"""Dev: fine-tune the last DINOv2-S blocks with a query-conditioned similarity objective on one CV fold (or all train).

python train_ft.py TAG FOLD [key=value ...]     FOLD in 0..3 (that fold is held out) or 'all'
Saves cache/ft_w/{TAG}_f{FOLD}.pt holding only the trainable parameters.
"""
import json, sys, time, queue, threading
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_common import (HERE, GRAY_DIR, LAB_DIR, W_DIR, P, TILE_P, Backbone, device, fold_assign, stem)  # noqa: E402
from devdata import examples  # noqa: E402

DEFAULT = dict(steps=1500, bs=6, lr=2e-5, wd=0.0, nblk=4, real_w=0.5, seed=0, p05=0.5, qtile=21, l2init=0.0,
               eval_every=250, warm=0.05, robust=1)
NQMAX = 400


def cells_in_box(box, c, gh, gw):
    """Same rule as pixfeat._cells: cell index ranges whose centres fall inside the box."""
    x0, y0, x1, y1 = box
    iy = np.arange(int(y0 // c), int(np.ceil(y1 / c))); ix = np.arange(int(x0 // c), int(np.ceil(x1 / c)))
    cy, cx = (iy + 0.5) * c, (ix + 0.5) * c
    iy2, ix2 = iy[(cy >= y0) & (cy < y1)], ix[(cx >= x0) & (cx < x1)]
    iy = iy2 if len(iy2) else iy
    ix = ix2 if len(ix2) else ix
    return np.clip(iy, 0, gh - 1), np.clip(ix, 0, gw - 1)


def crop_tile(gray, y0, x0, n, c, scale):
    """n x n cells starting at cell (y0, x0); native crop padded with paper, rescaled like features.dense_features."""
    H, W = gray.shape
    ys, xs = y0 * c, x0 * c
    a = np.asarray(gray[ys:min(ys + n * c, H), xs:min(xs + n * c, W)])
    out = np.full((n * c, n * c), 255, np.uint8)
    out[:a.shape[0], :a.shape[1]] = a
    if scale != 1.0:
        out = np.asarray(Image.fromarray(out).resize((n * P, n * P), Image.BILINEAR))
    return out.astype(np.float32) / 255.0


class Sampler:
    def __init__(self, exs, cfg, seed):
        self.exs, self.cfg = exs, cfg
        self.rng = np.random.default_rng(seed)
        self.docs = {}
        for i, e in enumerate(exs):
            self.docs.setdefault((e["kind"], e["document_id"]), []).append(i)
        self.kd = {k: [d for d in self.docs if d[0] == k] for k in ("real", "generated_cad")}
        self.cells = []
        for e in exs:
            lab = np.load(LAB_DIR / f"{e['id']}.npz")
            per = {}
            for key in ("l14", "l28"):
                l = lab[key].astype(np.float32); k, p = l[..., 0], l[..., 1]
                per[key] = (np.argwhere((k > 0) & (p >= 0.5 * k)), np.argwhere((k > 0) & (p < 0.5 * k)), l.shape[:2])
            self.cells.append(per)

    def pick(self):
        kind = "real" if self.rng.random() < self.cfg["real_w"] or not self.kd["generated_cad"] else "generated_cad"
        docs = self.kd[kind]
        return self.rng.choice(self.docs[docs[self.rng.integers(len(docs))]])

    def item(self):
        i = self.pick(); e = self.exs[i]
        scale = 0.5 if self.rng.random() < self.cfg["p05"] else 1.0
        c = int(round(P / scale)); key = "l14" if c == 14 else "l28"
        pos, neg, (gh, gw) = self.cells[i][key]
        pool = pos if (len(pos) and (self.rng.random() < 0.5 or not len(neg))) else neg
        cy, cx = pool[self.rng.integers(len(pool))]
        T = TILE_P
        y0 = int(np.clip(cy - self.rng.integers(T), 0, max(gh - T, 0)))
        x0 = int(np.clip(cx - self.rng.integers(T), 0, max(gw - T, 0)))
        gray = np.load(GRAY_DIR / f"{stem(e)}.npy", mmap_mode="r")
        ctx = crop_tile(gray, y0, x0, T, c, scale)
        lab = np.zeros((T, T, 2), np.float32)
        l = np.load(LAB_DIR / f"{e['id']}.npz")[key].astype(np.float32)
        sub = l[y0:y0 + T, x0:x0 + T]
        lab[:sub.shape[0], :sub.shape[1]] = sub
        # query tile centred on the box; query cells are those whose centres fall in the box
        iy, ix = cells_in_box(e["query_box"], c, gh, gw)
        bh, bw = iy.max() - iy.min() + 1, ix.max() - ix.min() + 1
        Tq = self.cfg["qtile"] if max(bh, bw) <= self.cfg["qtile"] - 8 else T
        qy0 = int(np.clip((iy.min() + iy.max() + 1) // 2 - Tq // 2, 0, max(gh - Tq, 0)))
        qx0 = int(np.clip((ix.min() + ix.max() + 1) // 2 - Tq // 2, 0, max(gw - Tq, 0)))
        qt = crop_tile(gray, qy0, qx0, Tq, c, scale)
        qmask = np.zeros((Tq, Tq), bool)
        qi = np.clip(iy - qy0, 0, Tq - 1); qj = np.clip(ix - qx0, 0, Tq - 1)
        qmask[np.ix_(qi, qj)] = True
        k = self.rng.integers(4); flip = self.rng.random() < 0.5
        aug = lambda a: (np.rot90(a, k)[:, ::-1] if flip else np.rot90(a, k))
        return np.ascontiguousarray(aug(ctx)), np.ascontiguousarray(aug(lab)), np.ascontiguousarray(aug(qt)), np.ascontiguousarray(aug(qmask))

    def batch(self):
        items = [self.item() for _ in range(self.cfg["bs"])]
        ctx = torch.from_numpy(np.stack([it[0] for it in items]))[:, None]
        lab = torch.from_numpy(np.stack([it[1] for it in items]))
        qts = [(torch.from_numpy(it[2])[None, None], torch.from_numpy(it[3])) for it in items]
        return ctx, lab, qts


def batches(sampler, n_workers=3):
    qq = queue.Queue(maxsize=4)
    subs = []
    for j in range(n_workers):
        s = Sampler.__new__(Sampler); s.__dict__.update(sampler.__dict__)
        s.rng = np.random.default_rng(sampler.rng.integers(1 << 31) + j); subs.append(s)

    def work(s):
        while True:
            qq.put(s.batch())
    for s in subs:
        threading.Thread(target=work, args=(s,), daemon=True).start()
    while True:
        yield qq.get()


class Head(torch.nn.Module):
    """Learned affine on the two fixed similarities the downstream features use (mean-proto cosine, top-3 cell cosine)."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([10.0, 10.0]))
        self.b = torch.nn.Parameter(torch.tensor([-5.0, -5.0]))


def query_protos(qtok, qmask, robust):
    """qtok (1,N,C) tokens of the query tile, qmask (Tq,Tq) -> normalized query cells (n,C) and prototype (C,)."""
    cells = F.normalize(qtok[0][qmask.reshape(-1)], dim=-1)
    if len(cells) > NQMAX:
        cells = cells[torch.randperm(len(cells), device=cells.device)[:NQMAX]]
    mean = F.normalize(cells.mean(0), dim=-1)
    if robust and len(cells) >= 3:
        ss = cells @ mean
        keep = ss >= torch.quantile(ss, 0.34)
        mean = F.normalize(cells[keep].mean(0), dim=-1)
    return cells, mean


def sims(tok, cells, proto):
    """tok (N,C) context tokens -> (N,2): cosine to prototype, mean top-3 cosine to query cells."""
    t = F.normalize(tok, dim=-1)
    s1 = t @ proto
    cs = t @ cells.T
    s2 = cs.topk(min(3, cs.shape[1]), dim=1).values.mean(1)
    return torch.stack([s1, s2], -1)


def item_loss(logit, lab):
    """Balanced BCE on known cells; logit (N,2), lab (N,2) known/positive fractions."""
    k, p = lab[:, 0:1], lab[:, 1:2]
    wp, wn = p, (k - p).clamp(min=0)
    lp, ln = F.softplus(-logit), F.softplus(logit)
    tp = (lp * wp).sum(0) / wp.sum().clamp(min=1e-6)
    tn = (ln * wn).sum(0) / wn.sum().clamp(min=1e-6)
    both = (wp.sum() > 0).float() + (wn.sum() > 0).float()
    return ((tp + tn) / both.clamp(min=1)).mean()


def step_loss(net, head, ctx, lab, qts, robust, dev):
    tok = net(ctx.to(dev))  # (B, N, C)
    losses = []
    for b, (qt, qm) in enumerate(qts):
        cells, proto = query_protos(net(qt.to(dev)), qm.to(dev), robust)
        s = sims(tok[b], cells, proto)
        losses.append(item_loss(s * head.a + head.b, lab[b].reshape(-1, 2).to(dev)))
    return torch.stack(losses).mean()


def train(exs, cfg, log, held=None):
    torch.manual_seed(cfg["seed"])
    dev = device()
    net = Backbone(cfg["nblk"]).to(dev)
    head = Head().to(dev)
    init = {k: v.clone() for k, v in net.trainable_state().items()} if cfg["l2init"] else None
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": cfg["lr"], "weight_decay": cfg["wd"]},
                             {"params": head.parameters(), "lr": 1e-2, "weight_decay": 0.0}])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [cfg["lr"], 1e-2], total_steps=cfg["steps"], pct_start=cfg["warm"])
    gen = batches(Sampler(exs, cfg, cfg["seed"]))
    hs = Sampler(held, cfg, 123) if held else None
    hb = [hs.batch() for _ in range(16)] if held else []
    if hb:
        print(f"step 0 heldout {heldout(net, head, hb, cfg, dev)}", file=log, flush=True)
    t, run = time.time(), []
    for step in range(cfg["steps"]):
        ctx, lab, qts = next(gen)
        net.train()
        loss = step_loss(net, head, ctx, lab, qts, cfg["robust"], dev)
        if init:
            reg = sum(((v - init[k].to(dev)) ** 2).sum() for k, v in net.m.state_dict().items() if k in init)
            loss = loss + cfg["l2init"] * reg
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        run.append(loss.item())
        if (step + 1) % 50 == 0:
            print(f"step {step+1} loss {np.mean(run[-50:]):.4f} a {head.a.data.cpu().numpy().round(2)} b {head.b.data.cpu().numpy().round(2)} {time.time()-t:.0f}s", file=log, flush=True)
        if hb and (step + 1) % cfg["eval_every"] == 0:
            print(f"step {step+1} heldout {heldout(net, head, hb, cfg, dev)}", file=log, flush=True)
    return net.eval(), head


@torch.no_grad()
def heldout(net, head, hb, cfg, dev):
    """Held-out loss plus rank AUC of each raw similarity over known cells (majority label), for the drift check."""
    net.eval(); losses, S, Yl = [], [], []
    for ctx, lab, qts in hb:
        tok = net(ctx.to(dev))
        for b, (qt, qm) in enumerate(qts):
            cells, proto = query_protos(net(qt.to(dev)), qm.to(dev), cfg["robust"])
            s = sims(tok[b], cells, proto); l = lab[b].reshape(-1, 2).to(dev)
            losses.append(item_loss(s * head.a + head.b, l).item())
            k, p = l[:, 0], l[:, 1]; known = k > 0
            S.append(s[known].cpu().numpy()); Yl.append((p[known] >= 0.5 * k[known]).cpu().numpy())
    S, Yl = np.concatenate(S), np.concatenate(Yl)
    from sklearn.metrics import roc_auc_score
    auc = [roc_auc_score(Yl, S[:, j]) for j in range(2)]
    return f"loss {np.mean(losses):.4f} auc {auc[0]:.4f}/{auc[1]:.4f} n {len(Yl)} pos {Yl.mean():.2f}"


def main():
    tag, fold = sys.argv[1], sys.argv[2]
    cfg = dict(DEFAULT)
    for kv in sys.argv[3:]:
        k, v = kv.split("="); cfg[k] = type(DEFAULT[k])(float(v)) if isinstance(DEFAULT[k], int) else float(v)
    tr = examples("train")
    if fold == "all":
        exs, held = tr, None
    else:
        qf = fold_assign()
        exs = [e for e, f in zip(tr, qf) if f != int(fold)]
        held = [e for e, f in zip(tr, qf) if f == int(fold) and e["kind"] == "real"]
    W_DIR.mkdir(parents=True, exist_ok=True)
    log = open(HERE / "logs" / f"{tag}_f{fold}.log", "w")
    print(json.dumps(cfg), file=log, flush=True)
    net, head = train(exs, cfg, log, held)
    torch.save({"cfg": cfg, "state": net.trainable_state(), "head": head.state_dict()}, W_DIR / f"{tag}_f{fold}.pt")
    print("done", file=log, flush=True)


if __name__ == "__main__":
    main()
