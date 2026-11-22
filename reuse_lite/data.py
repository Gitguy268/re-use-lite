"""VoiceBank-DEMAND crops with extra degradations, stored as fixed-size buckets per sample rate.

Every bucket holds crops of the same number of samples (default 24000). Frames per second are
the same at every rate (the STFT hop scales with the rate) and frequency bins grow with the rate,
so 24000 samples costs about the same at 8, 16, 24 and 48 kHz (3, 1.5, 1 and 0.5 seconds).
"""
import io
import json
import os

import numpy as np
import soundfile as sf
from scipy import signal

RATES = (8000, 16000, 24000, 48000)
RATE_P = (0.25, 0.45, 0.18, 0.12)
SRC_SR = 16000


def decode(b):
    x, sr = sf.read(io.BytesIO(b), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == SRC_SR, sr
    return x


def resample(x, sr):
    if sr == SRC_SR:
        return x.astype(np.float32)
    g = np.gcd(sr, SRC_SR)
    return signal.resample_poly(x, sr // g, SRC_SR // g).astype(np.float32)


def reverb(x, rng):
    """Exponential-decay noise RIR with a direct path, RMS kept. 16 kHz input."""
    rt60 = float(rng.uniform(0.2, 1.0))
    t = np.arange(int(rt60 * SRC_SR)) / SRC_SR
    rir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60) * float(rng.uniform(0.05, 0.4))
    rir[0] = 1.0
    rms = np.sqrt(np.mean(x ** 2)) + 1e-8
    y = signal.fftconvolve(x, rir)[:len(x)].astype(np.float32)
    return y * (rms / (np.sqrt(np.mean(y ** 2)) + 1e-8)), round(rt60, 3)


def lowpass(x, rng):
    fc = float(rng.uniform(1500, 6000))
    sos = signal.butter(8, fc, "low", fs=SRC_SR, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32), round(fc)


def clip(x, rng):
    th = float(np.max(np.abs(x)) * rng.uniform(0.1, 0.5))
    return np.clip(x, -th, th), round(th, 5)


def degrade(noisy, clean, rng, samples):
    """Return (noisy_crop, clean_crop, sr, flags) for one variant. Inputs are 16 kHz."""
    n = len(noisy)
    # pick a rate whose crop fits in the clip; 16 kHz always fits (padded if needed)
    sr = int(rng.choice(RATES, p=RATE_P))
    need = int(np.ceil(samples * SRC_SR / sr))
    if need > n and sr < SRC_SR:
        sr, need = SRC_SR, samples
    flags = {"sr": sr}
    if need <= n:
        s0 = int(rng.integers(0, n - need + 1))
        x, c = noisy[s0:s0 + need].copy(), clean[s0:s0 + need].copy()
    else:  # short clip: zero-pad at the end
        x, c = np.zeros(need, np.float32), np.zeros(need, np.float32)
        x[:n], c[:n] = noisy, clean
        flags["padded"] = n
    if rng.random() < 0.15:  # clean input: the student must learn to leave speech alone
        x = c.copy()
        flags["clean_in"] = True
    if rng.random() < 0.3:  # synthetic room: exponential-decay noise RIR with a direct path
        x, flags["rt60"] = reverb(x, rng)
    if rng.random() < 0.25:  # bandwidth limit
        x, flags["lowpass"] = lowpass(x, rng)
    peak = np.max(np.abs(x)) + 1e-8
    g = 10 ** (float(rng.uniform(-25, -1)) / 20) / peak
    x, c = x * g, c * g
    if rng.random() < 0.15:  # hard clipping
        x, flags["clip"] = clip(x, rng)
    x, c = resample(x, sr), resample(c, sr)
    x, c = _fit(x, samples), _fit(c, samples)
    # the clean target is only valid where the teacher would not add bandwidth above 8 kHz
    flags["clean_ok"] = sr <= SRC_SR
    return x, c, sr, flags


def _fit(x, samples):
    if len(x) >= samples:
        return x[:samples]
    return np.pad(x, (0, samples - len(x)))


class Buckets:
    """Read side of a prepared split: <root>/sr<rate>/{noisy,clean,teacher}.f32 + meta.jsonl + info.json.

    Each .f32 file is raw little-endian float32, shape (rows, samples)."""

    def __init__(self, root):
        self.root = root
        self.b = {}
        for sr in RATES:
            d = os.path.join(root, f"sr{sr}")
            if not os.path.isfile(os.path.join(d, "info.json")):
                continue
            info = json.load(open(os.path.join(d, "info.json")))
            meta = [json.loads(l) for l in open(os.path.join(d, "meta.jsonl"))]
            assert len(meta) == info["rows"], d
            arr = {}
            for k in ("noisy", "clean", "teacher"):
                f = os.path.join(d, f"{k}.f32")
                if os.path.isfile(f) and os.path.getsize(f) == 4 * info["rows"] * info["samples"]:
                    arr[k] = np.memmap(f, np.float32, "r", shape=(info["rows"], info["samples"]))
            self.b[sr] = dict(meta=meta, info=info, **arr)

    def sizes(self):
        return {sr: len(v["meta"]) for sr, v in self.b.items()}
