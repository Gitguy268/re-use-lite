# SPDX-License-Identifier: Apache-2.0
"""Fused selective scan for Apple GPUs (torch.mps.compile_shader), forward + backward.

y[b,t,d] = sum_n C[b,t,n] h[b,t,d,n] + D[d] u[b,t,d]
h[b,t,d,n] = exp(delta[b,t,d] A[d,n]) h[b,t-1,d,n] + delta[b,t,d] B[b,t,n] u[b,t,d]

One thread per (b, d); the n-state (<= 16) lives in registers.
"""
import torch

NMAX = 16
_SRC = r"""
#include <metal_stdlib>
using namespace metal;
#define NMAX 16

kernel void scan_fwd(device const float* u     [[buffer(0)]],
                     device const float* delta [[buffer(1)]],
                     device const float* A     [[buffer(2)]],
                     device const float* B     [[buffer(3)]],
                     device const float* C     [[buffer(4)]],
                     device const float* Dv    [[buffer(5)]],
                     device float* y           [[buffer(6)]],
                     device float* H           [[buffer(7)]],
                     constant uint& L          [[buffer(8)]],
                     constant uint& Dm         [[buffer(9)]],
                     constant uint& N          [[buffer(10)]],
                     constant uint& save       [[buffer(11)]],
                     uint tid [[thread_position_in_grid]]) {
    uint b = tid / Dm, d = tid % Dm;
    float h[NMAX]; float a[NMAX];
    for (uint n = 0; n < N; n++) { h[n] = 0.0f; a[n] = A[d * N + n]; }
    float Dd = Dv[d];
    for (uint t = 0; t < L; t++) {
        uint i = (b * L + t) * Dm + d;
        float dt = delta[i], ut = u[i], acc = 0.0f;
        uint bn = (b * L + t) * N;
        for (uint n = 0; n < N; n++) {
            h[n] = exp(dt * a[n]) * h[n] + dt * B[bn + n] * ut;
            acc += C[bn + n] * h[n];
            if (save) H[i * N + n] = h[n];
        }
        y[i] = acc + Dd * ut;
    }
}

kernel void scan_bwd(device const float* u     [[buffer(0)]],
                     device const float* delta [[buffer(1)]],
                     device const float* A     [[buffer(2)]],
                     device const float* B     [[buffer(3)]],
                     device const float* C     [[buffer(4)]],
                     device const float* Dv    [[buffer(5)]],
                     device const float* H     [[buffer(6)]],
                     device const float* gy    [[buffer(7)]],
                     device float* gu          [[buffer(8)]],
                     device float* gdelta      [[buffer(9)]],
                     device float* gA          [[buffer(10)]],   // (b, d, n) partial
                     device float* gB          [[buffer(11)]],   // (b, t, d, n) partial
                     device float* gC          [[buffer(12)]],   // (b, t, d, n) partial
                     device float* gD          [[buffer(13)]],   // (b, d) partial
                     constant uint& L          [[buffer(14)]],
                     constant uint& Dm         [[buffer(15)]],
                     constant uint& N          [[buffer(16)]],
                     uint tid [[thread_position_in_grid]]) {
    uint b = tid / Dm, d = tid % Dm;
    float gh[NMAX]; float a[NMAX]; float ga[NMAX];
    for (uint n = 0; n < N; n++) { gh[n] = 0.0f; a[n] = A[d * N + n]; ga[n] = 0.0f; }
    float Dd = Dv[d], gDd = 0.0f;
    for (int t = int(L) - 1; t >= 0; t--) {
        uint i = (b * L + uint(t)) * Dm + d;
        uint bn = (b * L + uint(t)) * N;
        float dt = delta[i], ut = u[i], g = gy[i];
        gDd += g * ut;
        float gut = g * Dd, gdt = 0.0f;
        for (uint n = 0; n < N; n++) {
            float hn = H[i * N + n];
            float hp = (t > 0) ? H[(i - Dm) * N + n] : 0.0f;
            gC[i * N + n] = g * hn;
            gh[n] += g * C[bn + n];
            float dA = exp(dt * a[n]);
            float gdA = gh[n] * hp;
            gdt += gdA * a[n] * dA + gh[n] * B[bn + n] * ut;
            ga[n] += gdA * dt * dA;
            gB[i * N + n] = gh[n] * dt * ut;
            gut += gh[n] * dt * B[bn + n];
            gh[n] *= dA;
        }
        gu[i] = gut; gdelta[i] = gdt;
    }
    for (uint n = 0; n < N; n++) gA[(b * Dm + d) * N + n] = ga[n];
    gD[b * Dm + d] = gDd;
}
"""
_lib = None


def _get():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_SRC)
    return _lib


class _Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D):
        b, l, d = u.shape
        n = A.shape[1]
        assert n <= NMAX
        u, delta, A, B, C, D = (t.contiguous().float() for t in (u, delta, A, B, C, D))
        save = ctx.needs_input_grad and any(ctx.needs_input_grad)
        y = torch.empty_like(u)
        H = torch.empty(b, l, d, n, device=u.device) if save else torch.empty(1, device=u.device)
        _get().scan_fwd(u, delta, A, B, C, D, y, H, l, d, n, int(bool(save)), threads=b * d)
        if save:
            ctx.save_for_backward(u, delta, A, B, C, D, H)
        return y

    @staticmethod
    def backward(ctx, gy):
        u, delta, A, B, C, D, H = ctx.saved_tensors
        b, l, d = u.shape
        n = A.shape[1]
        gy = gy.contiguous().float()
        gu, gdelta = torch.empty_like(u), torch.empty_like(u)
        gA = torch.empty(b, d, n, device=u.device)
        gB = torch.empty(b, l, d, n, device=u.device)
        gC = torch.empty(b, l, d, n, device=u.device)
        gD = torch.empty(b, d, device=u.device)
        _get().scan_bwd(u, delta, A, B, C, D, H, gy, gu, gdelta, gA, gB, gC, gD, l, d, n, threads=b * d)
        return gu, gdelta, gA.sum(0), gB.sum(2), gC.sum(2), gD.sum(0)


def selective_scan_metal(u, delta, A, B, C, D):
    return _Scan.apply(u, delta, A, B, C, D)
