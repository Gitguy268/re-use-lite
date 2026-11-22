"""Enhance every .wav in a folder with an MLX model folder (runs in the mlx-audio venv).

usage: mlx_enhance.py --model out/mlx --inp DIR --out DIR   (whole file, no chunking, float32)"""
import argparse
import os

import numpy as np
import soundfile as sf
from mlx_audio.sts import load

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
model = load(a.model, strict=True)
assert len(model.TSMamba) == model.config.num_tfmamba, "block count mismatch"
os.makedirs(a.out, exist_ok=True)
for f in sorted(os.listdir(a.inp)):
    if not f.endswith(".wav") or f.startswith("._"):
        continue
    x, sr = sf.read(os.path.join(a.inp, f), dtype="float32")
    r = model.generate(x, sample_rate=sr, chunk_seconds=None)
    y = np.array(r.audio, dtype=np.float32)
    assert r.sample_rate == sr and y.shape[0] == x.shape[0], (f, r.sample_rate, y.shape, x.shape)
    sf.write(os.path.join(a.out, f), y, sr, subtype="FLOAT")
print(f"mlx: {a.out} done ({len(model.TSMamba)} blocks)")
