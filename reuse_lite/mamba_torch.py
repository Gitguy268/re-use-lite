# SPDX-License-Identifier: Apache-2.0
# Pure-PyTorch Mamba (v1) block, parameter-compatible with mamba_ssm.Mamba.
# Forward math follows the reference (slow) path of mamba_ssm/modules/mamba_simple.py
# and selective_scan_ref in mamba_ssm/ops/selective_scan_interface.py
# (Copyright (c) 2023 Tri Dao, Albert Gu; Apache License 2.0). See NOTICE.
# Runs on CPU, CUDA and Apple MPS. The scan is chunked: inside a chunk it is
# computed in parallel in log space (logcumsumexp), state is carried between chunks.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan_seq(u, delta, A, B, C, D):
    """Sequential reference. u, delta: (b, l, d); A: (d, n); B, C: (b, l, n); D: (d,)."""
    b, l, d = u.shape
    h = u.new_zeros(b, d, A.shape[1])
    ys = []
    for t in range(l):
        dA = torch.exp(delta[:, t, :, None] * A)
        h = dA * h + delta[:, t, :, None] * B[:, t, None, :] * u[:, t, :, None]
        ys.append((h * C[:, t, None, :]).sum(-1))
    return torch.stack(ys, 1) + u * D


def _lcse(x, dim):
    # logcumsumexp that also works where the MPS kernel is missing
    try:
        return torch.logcumsumexp(x, dim)
    except (NotImplementedError, RuntimeError):
        return torch.logcumsumexp(x.cpu(), dim).to(x.device)


def selective_scan_chunked(u, delta, A, B, C, D, chunk=32):
    """Same result as selective_scan_seq.

    Inside a chunk, h_t = exp(S_t) * (h0 + sum_{s<=t} exp(-S_s) v_s) with S = cumsum(delta*A),
    v = delta*B*u. The sum is done as logcumsumexp over the positive and negative parts of v,
    so nothing overflows.
    """
    b, l, d = u.shape
    n = A.shape[1]
    h = u.new_zeros(b, d, n)
    ys = []
    tiny = torch.finfo(u.dtype).tiny
    for s0 in range(0, l, chunk):
        dl = delta[:, s0:s0 + chunk]                      # (b,k,d)
        uu = u[:, s0:s0 + chunk]
        Bc = B[:, s0:s0 + chunk]                           # (b,k,n)
        Cc = C[:, s0:s0 + chunk]
        dA = dl[..., None] * A                             # (b,k,d,n)  <= 0
        S = torch.cumsum(dA, 1)
        v = (dl * uu)[..., None] * Bc[:, :, None, :]       # (b,k,d,n)
        lp = torch.log(v.clamp_min(0) + tiny) - S
        ln = torch.log((-v).clamp_min(0) + tiny) - S
        hk = torch.exp(S + _lcse(lp, 1)) - torch.exp(S + _lcse(ln, 1))
        hk = hk + torch.exp(S) * h[:, None]
        ys.append((hk * Cc[:, :, None, :]).sum(-1))
        h = hk[:, -1]
    return torch.cat(ys, 1) + u * D


def selective_scan(u, delta, A, B, C, D):
    """Fused Metal kernel on Apple GPUs; plain sequential loop elsewhere (fastest on CPU)."""
    if u.device.type == "mps":
        from reuse_lite.scan_metal import selective_scan_metal
        return selective_scan_metal(u, delta, A, B, C, D)
    return selective_scan_seq(u, delta, A, B, C, D)


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, conv_bias=True, bias=False,
                 chunk=32, **_):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.chunk = chunk
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # init as in mamba_simple (only matters for training from scratch)
        dt_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_std, dt_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states):
        _, l, _ = hidden_states.shape
        x, z = self.in_proj(hidden_states).chunk(2, dim=-1)           # (b,l,d_inner)
        x = self.conv1d(x.transpose(1, 2))[..., :l].transpose(1, 2)
        x = F.silu(x)
        dt, Bm, Cm = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], -1)
        dt = F.softplus(self.dt_proj(dt))                              # bias inside, as delta_bias
        A = -torch.exp(self.A_log.float())
        y = selective_scan(x, dt, A, Bm, Cm, self.D.float())
        y = y * F.silu(z)
        return self.out_proj(y)
