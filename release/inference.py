# SPDX-License-Identifier: OpenMDW-1.1
"""RE-USE Lite: enhance speech files (8-48 kHz) on CUDA, Apple GPU (MPS) or CPU.

usage: python inference.py input.wav [more.wav | folder ...] -o out_dir [--device auto|cuda|mps|cpu]
                           [--chunk-seconds 10 --overlap-seconds 1]

The output keeps the input sample rate, channel count and length. Each channel is enhanced on its
own. Long files are processed in Hann-weighted overlapping chunks (default above 30 s), the same
way as NVIDIA's inference_chunk.py; chunking changes the bidirectional context slightly.
Uses the CUDA mamba_ssm package when it imports, otherwise the bundled pure-PyTorch Mamba.
"""
import argparse
import json
import math
import os
import sys
import types

import numpy as np
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
try:
    import mamba_ssm  # noqa: F401
except Exception:
    from reuse_lite.mamba_torch import Mamba
    sys.modules["mamba_ssm"] = types.SimpleNamespace(Mamba=Mamba)
from safetensors.torch import load_file  # noqa: E402
from models.generator_SEMamba_time_d4 import SEMamba  # noqa: E402
from models.stfts import mag_phase_istft, mag_phase_stft  # noqa: E402

EXTS = (".wav", ".flac", ".ogg", ".aiff", ".aif")


def make_even(v):
    v = int(round(v))
    return v if v % 2 == 0 else v + 1


def load(model_dir=ROOT, device="cpu"):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    m = SEMamba(cfg)
    m.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    return m.to(device).eval(), cfg


@torch.inference_mode()
def enhance(model, cfg, wav, sr):
    """wav: (1, n) float tensor on the model's device -> (1, n)."""
    st = cfg["stft_cfg"]
    n_fft, hop, win = (make_even(st[k] * sr // st["sampling_rate"]) for k in ("n_fft", "hop_size", "win_size"))
    cf = cfg["model_cfg"]["compress_factor"]
    mag, pha, _ = mag_phase_stft(wav, n_fft, hop, win, compress_factor=cf, center=True, addeps=False)
    amp_g, pha_g, _ = model(mag, pha)
    m = torch.expm1(torch.relu(amp_g))
    zero_portion = torch.sum(m == 0, 1) / m.shape[1]
    amp_g = amp_g.masked_fill((zero_portion > 0.5)[:, None, :], 0.0)
    out = mag_phase_istft(amp_g, pha_g, n_fft, hop, win, cf)
    n = wav.shape[1]
    if out.shape[1] >= n:
        return out[:, :n]
    return torch.nn.functional.pad(out, (0, n - out.shape[1]), value=1e-8)


def enhance_long(model, cfg, x, sr, device, chunk_s, overlap_s):
    """x: (n,) numpy mono. Whole file when it is short enough, otherwise overlapping chunks."""
    n = len(x)
    if chunk_s <= 0 or n <= int(chunk_s * sr):
        return enhance(model, cfg, torch.from_numpy(x)[None].to(device), sr)[0].float().cpu().numpy()
    size, hop = int(chunk_s * sr), int((chunk_s - overlap_s) * sr)
    assert 0 < hop <= size, "overlap must be shorter than the chunk"
    win = np.clip(np.hanning(size), 1e-7, None).astype(np.float32)
    out, wsum = np.zeros(n, np.float32), np.zeros(n, np.float32)
    starts = list(range(0, n - size, hop)) + [n - size]
    for s in starts:
        y = enhance(model, cfg, torch.from_numpy(np.ascontiguousarray(x[s:s + size]))[None].to(device), sr)
        out[s:s + size] += y[0].float().cpu().numpy() * win
        wsum[s:s + size] += win
    return out / wsum


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default="enhanced")
    ap.add_argument("--model", default=ROOT, help="folder with config.json + model.safetensors")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--chunk-seconds", type=float, default=None, help="default: 10 for files over 30 s, else whole file")
    ap.add_argument("--overlap-seconds", type=float, default=1.0)
    a = ap.parse_args()
    dev = pick_device(a.device)
    model, cfg = load(a.model, dev)
    files = []
    for p in a.inputs:
        if os.path.isdir(p):
            files += sorted(os.path.join(r, f) for r, _, fs in os.walk(p) for f in fs
                            if f.lower().endswith(EXTS) and not f.startswith("._"))
        else:
            files.append(p)
    os.makedirs(a.out, exist_ok=True)
    for f in files:
        x, sr = sf.read(f, dtype="float32", always_2d=True)
        if not 8000 <= sr <= 48000:
            print(f"skip {f}: {sr} Hz is outside 8-48 kHz"); continue
        if x.shape[0] <= sr // 50:
            print(f"skip {f}: shorter than 20 ms"); continue
        chunk = a.chunk_seconds if a.chunk_seconds is not None else (10.0 if x.shape[0] > 30 * sr else 0)
        y = np.stack([enhance_long(model, cfg, np.ascontiguousarray(x[:, c]), sr, dev, chunk, a.overlap_seconds)
                      for c in range(x.shape[1])], 1)
        dst = os.path.join(a.out, os.path.splitext(os.path.basename(f))[0] + "_enhanced.wav")
        sf.write(dst, np.clip(y, -1, 1), sr, subtype="FLOAT")
        print(f"{f} -> {dst} ({x.shape[0] / sr:.1f} s, {sr} Hz, {x.shape[1]} ch, {dev})")


if __name__ == "__main__":
    main()
