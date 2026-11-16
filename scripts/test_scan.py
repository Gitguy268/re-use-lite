"""Gate 0a: chunked scan == sequential scan, at time (~200 after the encoder's /4) and freq (~161) lengths."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from reuse_lite.mamba_torch import selective_scan_seq, selective_scan_chunked

torch.manual_seed(0)
for dev in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
    for L in (200, 161, 400, 323):
        b, d, n = 8, 256, 16
        u = torch.randn(b, L, d)
        delta = F.softplus(torch.randn(b, L, d) * 2 - 1)          # wide range incl. large steps
        A = -torch.exp(torch.randn(d, n) * 1.5)                    # |A| from ~0.01 to ~100
        B, C, D = torch.randn(b, L, n), torch.randn(b, L, n), torch.randn(d)
        ref = selective_scan_seq(u.double(), delta.double(), A.double(), B.double(), C.double(), D.double())
        args = [t.to(dev) for t in (u, delta, A, B, C, D)]
        t0 = time.time(); out = selective_scan_chunked(*args).cpu().double(); t1 = time.time()
        err = (out - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        print(f"{dev} L={L} max_abs_err={err:.2e} rel={rel:.2e} chunked {t1-t0:.3f}s")
        assert rel < 1e-4, "scan mismatch"
print("OK")
