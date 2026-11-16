"""Metal scan vs sequential reference: forward values, gradients, speed."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from reuse_lite.mamba_torch import selective_scan_seq
from reuse_lite.scan_metal import selective_scan_metal

torch.manual_seed(0)
b, L, d, n = 6, 57, 32, 16
mk = lambda *s: torch.randn(*s, dtype=torch.float64)
u, dl, A, B, C, D = mk(b, L, d), F.softplus(mk(b, L, d) - 1), -torch.exp(mk(d, n)), mk(b, L, n), mk(b, L, n), mk(d)
ins = [t.requires_grad_() for t in (u, dl, A, B, C, D)]
y = selective_scan_seq(*ins); gy = torch.randn_like(y)
ref_g = torch.autograd.grad(y, ins, gy)
mins = [t.detach().float().to("mps").requires_grad_() for t in ins]
ym = selective_scan_metal(*mins)
mg = torch.autograd.grad(ym, mins, gy.float().to("mps"))
print("fwd rel err", ((ym.cpu().double() - y).abs().max() / y.abs().max()).item())
for name, a, r in zip("u delta A B C D".split(), mg, ref_g):
    e = ((a.cpu().double() - r).abs().max() / r.abs().max()).item()
    print(f"grad {name} rel err {e:.2e}"); assert e < 1e-4, name

for (bb, LL) in [(161, 200), (200, 161), (442, 256)]:
    args = [torch.randn(bb, LL, 256), F.softplus(torch.randn(bb, LL, 256) - 1), -torch.exp(torch.randn(256, 16)),
            torch.randn(bb, LL, 16), torch.randn(bb, LL, 16), torch.randn(256)]
    args = [a.to("mps") for a in args]
    with torch.no_grad():
        selective_scan_metal(*args); torch.mps.synchronize(); t = time.time()
        for _ in range(5): selective_scan_metal(*args)
        torch.mps.synchronize()
    print(f"metal fwd b={bb} L={LL}: {(time.time()-t)/5*1000:.1f} ms")
print("OK")
