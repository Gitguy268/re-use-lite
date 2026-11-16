"""Compare scan implementations at realistic shapes (16 kHz, 4 s clip: time L=200 x 161 seqs)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from reuse_lite.mamba_torch import selective_scan_seq, selective_scan_chunked


def scan_segsum(u, delta, A, B, C, D, chunk=16):
    b, l, d = u.shape
    h = u.new_zeros(b, d, A.shape[1]); ys = []
    for s0 in range(0, l, chunk):
        dl, uu, Bc, Cc = delta[:, s0:s0+chunk], u[:, s0:s0+chunk], B[:, s0:s0+chunk], C[:, s0:s0+chunk]
        k = dl.shape[1]
        S = torch.cumsum(dl[..., None] * A, 1)                       # (b,k,d,n)
        mask = torch.tril(torch.ones(k, k, dtype=torch.bool, device=u.device))
        seg = (S[:, :, None] - S[:, None, :]).masked_fill(~mask[None, :, :, None, None], -float("inf"))
        W = torch.exp(seg)                                           # (b,t,s,d,n)
        v = (dl * uu)[..., None] * Bc[:, :, None, :]                 # (b,s,d,n)
        hk = torch.einsum("btsdn,bsdn->btdn", W, v) + torch.exp(S) * h[:, None]
        ys.append((hk * Cc[:, :, None, :]).sum(-1)); h = hk[:, -1]
    return torch.cat(ys, 1) + u * D


def run(fn, args, dev, reps=3):
    fn(*args)
    if dev == "mps": torch.mps.synchronize()
    t = time.time()
    for _ in range(reps): fn(*args)
    if dev == "mps": torch.mps.synchronize()
    return (time.time() - t) / reps


for dev in ["mps", "cpu"]:
    for (b, L) in [(161, 200), (200, 161)]:
        d, n = 256, 16
        g = torch.Generator().manual_seed(0)
        args = [torch.randn(b, L, d, generator=g), F.softplus(torch.randn(b, L, d, generator=g) - 1),
                -torch.exp(torch.randn(d, n, generator=g)), torch.randn(b, L, n, generator=g),
                torch.randn(b, L, n, generator=g), torch.randn(d, generator=g)]
        args = [a.to(dev) for a in args]
        r = {}
        for name, fn in [("seq", selective_scan_seq), ("log32", selective_scan_chunked),
                         ("log64", lambda *a: selective_scan_chunked(*a, chunk=64)),
                         ("seg8", lambda *a: scan_segsum(*a, chunk=8)), ("seg16", scan_segsum)]:
            with torch.no_grad():
                r[name] = run(fn, args, dev)
        print(dev, b, L, {k: f"{v*1000:.0f}ms" for k, v in r.items()}, flush=True)
