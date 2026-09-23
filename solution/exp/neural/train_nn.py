"""Dev: train the query-conditioned conv head on one CV fold (or all train) and cache its logit maps.

python train_nn.py TAG FOLD [key=value ...]    FOLD in 0..3 (predicts that fold's train queries) or 'all' (predicts val)
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nn_common import ROOT, HERE, IMG_DIR, Q_DIR, query_cells  # noqa: E402
from model import Net, DILS  # noqa: E402
sys.path.insert(0, str(ROOT / "solution"))
from devdata import examples  # noqa: E402

PRED = ROOT / "cache" / "nn_pred"
NORM_PATH = HERE / "weights" / "norm.npz"
NQMAX = 256
DEFAULT = dict(steps=3000, bs=8, crop=128, lr=1e-3, wd=0.05, d=64, fixed=1, in_drop=0.1, real_w=0.5, seed=0,
               absf=1, nblk=8, eval_every=0, ctx=0, resid=0, dmax=0.0, dl2=0.0)
GBM = ROOT / "cache" / "nn_gbm" / "tx_v1"
CTX = {"on": False}


def stem(e):
    return Path(e["image"]).stem[:16]


def fold_assign():
    exs = examples("train")
    docs = sorted({e["document_id"] for e in exs})
    rng = np.random.default_rng(0); rng.shuffle(docs)
    fo = {d: i % 4 for i, d in enumerate(docs)}
    return np.array([fo[e["document_id"]] for e in exs])


def norm_stats():
    if NORM_PATH.exists():
        return dict(np.load(NORM_PATH))
    rng = np.random.default_rng(0)
    samp = []
    for s in sorted({stem(e) for e in examples("train")}):
        a = np.load(IMG_DIR / f"{s}.npy", mmap_mode="r").reshape(-1, 132)
        samp.append(np.asarray(a[np.sort(rng.choice(len(a), 2000, replace=False))], np.float32))
    X = np.concatenate(samp)
    out = {"mu": X.mean(0), "sd": X.std(0) + 1e-3}
    np.savez(NORM_PATH, **out)
    return out


def ranks(sims, qs):
    return np.stack([np.interp(sims[..., c], qs[c], np.linspace(0, 1, 101)) for c in range(sims.shape[-1])], -1).astype(np.float32)


def gbm_channels(e, sl=(slice(None), slice(None))):
    """Clipped tx_v1 OOF logit / 4 and its within-image rank."""
    g = np.load(GBM / f"{e['id']}.npy").astype(np.float32)
    qs = np.quantile(g, np.linspace(0, 1, 101))
    c = g[sl]
    return np.stack([np.clip(c, -10, 10) / 4, np.interp(c, qs, np.linspace(0, 1, 101))], -1).astype(np.float32)


def load_query(e, norm):
    """Full-image normalized inputs for one query: x (gh,gw,132), sims (gh,gw,20), qtok (N,132)."""
    x = (np.load(IMG_DIR / f"{stem(e)}.npy").astype(np.float32) - norm["mu"]) / norm["sd"]
    q = np.load(Q_DIR / f"{e['id']}.npz")
    sims = q["sims"].astype(np.float32)
    sims = np.concatenate([sims, ranks(sims, q["qs"])] + ([gbm_channels(e)] if CTX["on"] else []), -1)
    gh, gw = x.shape[:2]
    y0, y1, x0, x1 = query_cells(e["query_box"], gh, gw)
    return x, sims, x[y0:y1, x0:x1].reshape(-1, x.shape[-1])


class Sampler:
    def __init__(self, exs, cfg, norm, seed):
        self.exs, self.cfg, self.norm = exs, cfg, norm
        self.rng = np.random.default_rng(seed)
        self.docs = {}
        for i, e in enumerate(exs):
            self.docs.setdefault((e["kind"], e["document_id"]), []).append(i)
        self.kd = {k: [d for d in self.docs if d[0] == k] for k in ("real", "generated_cad")}
        self.cells = []
        for e in exs:
            lab = np.load(Q_DIR / f"{e['id']}.npz")["lab"].astype(np.float32)
            k, p = lab[..., 0], lab[..., 1]
            self.cells.append((np.argwhere((k > 0) & (p >= 0.5 * k)), np.argwhere((k > 0) & (p < 0.5 * k)), lab.shape[:2]))

    def pick(self):
        kind = "real" if self.rng.random() < self.cfg["real_w"] or not self.kd["generated_cad"] else "generated_cad"
        docs = self.kd[kind]
        return self.rng.choice(self.docs[docs[self.rng.integers(len(docs))]])

    def item(self):
        C = self.cfg["crop"]
        i = self.pick(); e = self.exs[i]
        pos, neg, (gh, gw) = self.cells[i]
        pool = pos if (len(pos) and (self.rng.random() < 0.5 or not len(neg))) else neg
        cy, cx = pool[self.rng.integers(len(pool))]
        y0 = int(np.clip(cy - self.rng.integers(C), 0, max(gh - C, 0)))
        x0 = int(np.clip(cx - self.rng.integers(C), 0, max(gw - C, 0)))
        xm = np.load(IMG_DIR / f"{stem(e)}.npy", mmap_mode="r")
        n = self.norm
        x = (np.asarray(xm[y0:y0 + C, x0:x0 + C], np.float32) - n["mu"]) / n["sd"]
        q = np.load(Q_DIR / f"{e['id']}.npz")
        s = q["sims"][y0:y0 + C, x0:x0 + C].astype(np.float32)
        s = np.concatenate([s, ranks(s, q["qs"])] + ([gbm_channels(e, (slice(y0, y0 + C), slice(x0, x0 + C)))] if CTX["on"] else []), -1)
        lab = q["lab"][y0:y0 + C, x0:x0 + C].astype(np.float32)
        qy0, qy1, qx0, qx1 = query_cells(e["query_box"], gh, gw)
        qt = (np.asarray(xm[qy0:qy1, qx0:qx1], np.float32).reshape(-1, 132) - n["mu"]) / n["sd"]
        if len(qt) > NQMAX:
            qt = qt[self.rng.choice(len(qt), NQMAX, replace=False)]
        k = self.rng.integers(4)
        x, s, lab = (np.rot90(a, k) for a in (x, s, lab))
        if self.rng.random() < 0.5:
            x, s, lab = (a[:, ::-1] for a in (x, s, lab))
        h, w = x.shape[:2]
        pad = lambda a: np.pad(a, ((0, C - h), (0, C - w), (0, 0)))
        return pad(x), pad(s), pad(lab), qt

    def batch(self):
        items = [self.item() for _ in range(self.cfg["bs"])]
        N = max(len(it[3]) for it in items)
        qt = np.zeros((len(items), N, 132), np.float32); qm = np.zeros((len(items), N), bool)
        for j, it in enumerate(items):
            qt[j, :len(it[3])] = it[3]; qm[j, :len(it[3])] = True
        st = lambda k: torch.from_numpy(np.stack([it[k] for it in items]))
        return st(0), st(1), st(2), torch.from_numpy(qt), torch.from_numpy(qm)


def loss_fn(logit, lab):
    k, p = lab[..., 0], lab[..., 1]
    wp, wn = p, (k - p).clamp(min=0)
    lp = F.softplus(-logit); ln = F.softplus(logit)
    sp, sn = wp.sum((1, 2)), wn.sum((1, 2))
    tp = (lp * wp).sum((1, 2)) / sp.clamp(min=1e-6)
    tn = (ln * wn).sum((1, 2)) / sn.clamp(min=1e-6)
    both = (sp > 0).float() + (sn > 0).float()
    return ((tp + tn) / both.clamp(min=1)).mean()


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def batches(sampler, n_workers=3):
    """Background thread prefetch (numpy work releases the GIL for the heavy parts)."""
    import queue, threading
    qq = queue.Queue(maxsize=6)
    subs = [Sampler.__new__(Sampler) for _ in range(n_workers)]
    for j, s in enumerate(subs):
        s.__dict__.update(sampler.__dict__); s.rng = np.random.default_rng(sampler.rng.integers(1 << 31) + j)

    def work(s):
        while True:
            qq.put(s.batch())
    for s in subs:
        threading.Thread(target=work, args=(s,), daemon=True).start()
    while True:
        yield qq.get()


def build_net(cfg):
    c = dict(DEFAULT, **cfg)
    CTX["on"] = bool(c["ctx"])
    return Net(d=c["d"], use_fixed=bool(c["fixed"]), in_drop=c["in_drop"], absf=bool(c["absf"]), dils=DILS[:c["nblk"]],
               nsim=20 + 2 * CTX["on"], resid=bool(c["resid"]), dmax=c["dmax"])


def heldout_curve(net, norm, target):
    """Dev diagnostic: cell-level doc-macro IoU (known-fraction weighted) on held-out real queries."""
    net.eval(); docs = {}
    for e in target:
        if e["kind"] != "real":
            continue
        lab = np.load(Q_DIR / f"{e['id']}.npz")["lab"].astype(np.float32)
        k, p = lab[..., 0], lab[..., 1]
        if p.sum() == 0:
            continue
        pr = predict(net, norm, e) > 0
        tp = (p * pr).sum(); fp = ((k - p) * pr).sum(); fn = (p * ~pr).sum()
        docs.setdefault(e["document_id"], []).append(tp / (tp + fp + fn))
    net.train()
    return np.mean([np.mean(v) for v in docs.values()])


def train(exs, cfg, log, target=None):
    torch.manual_seed(cfg["seed"])
    norm = norm_stats()
    dev = device()
    net = build_net(cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["steps"], pct_start=0.05)
    gen = batches(Sampler(exs, cfg, norm, cfg["seed"]))
    t, run = time.time(), []
    for step in range(cfg["steps"]):
        x, s, lab, qt, qm = (a.to(dev) for a in next(gen))
        loss = loss_fn(net(x, s, qt, qm), lab)
        if cfg["dl2"]:
            loss = loss + cfg["dl2"] * (net.last_delta ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        run.append(loss.item())
        if (step + 1) % 100 == 0:
            print(f"step {step+1} loss {np.mean(run[-100:]):.4f} {time.time()-t:.0f}s", file=log, flush=True)
        if cfg["eval_every"] and target and (step + 1) % cfg["eval_every"] == 0:
            print(f"step {step+1} heldout cell doc-macro {heldout_curve(net, norm, target):.4f}", file=log, flush=True)
    return net.eval(), norm


@torch.no_grad()
def predict(net, norm, e):
    x, s, qt = load_query(e, norm)
    dev = next(net.parameters()).device
    tt = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32))[None].to(dev)
    qm = torch.ones(1, len(qt), dtype=torch.bool, device=dev)
    return net(tt(x), tt(s), tt(qt), qm)[0].float().cpu().numpy()


def main():
    tag, fold = sys.argv[1], sys.argv[2]
    cfg = dict(DEFAULT)
    for kv in sys.argv[3:]:
        k, v = kv.split("="); cfg[k] = type(DEFAULT[k])(float(v)) if isinstance(DEFAULT[k], int) else float(v)
    tr = examples("train")
    if fold == "all":
        exs, target = tr, examples("val")
    else:
        qf = fold_assign()
        exs = [e for e, f in zip(tr, qf) if f != int(fold)]
        target = [e for e, f in zip(tr, qf) if f == int(fold)]
    log = open(HERE / "logs" / f"{tag}_f{fold}.log", "w")
    print(json.dumps(cfg), file=log, flush=True)
    net, norm = train(exs, cfg, log, target if fold != "all" else None)
    torch.save({"cfg": cfg, "state": net.state_dict()}, HERE / "weights" / f"{tag}_f{fold}.pt")
    out = PRED / tag; out.mkdir(parents=True, exist_ok=True)
    for e in target:
        np.save(out / f"{e['id']}.npy", predict(net, norm, e).astype(np.float16))
    print("done", file=log, flush=True)


if __name__ == "__main__":
    main()
