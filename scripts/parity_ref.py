"""Teacher reference outputs on CPU PyTorch, for the CUDA (mamba_ssm) parity probe on Modal.

usage: parity_ref.py --out results/parity_ref.npz
Clips: mic_test2 (vendor sample), two VoiceBank-DEMAND test clips at 16 kHz and one at 8 kHz.
"""
import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("REUSE_TORCH_MAMBA", "1")
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

from reuse_lite.data import resample
from reuse_lite.model import enhance, load_teacher

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()

clips = []
x, sr = sf.read("vendor/noisy_audio/mic_test2.wav", dtype="float32")
clips.append(("mic_test2", (x.mean(1) if x.ndim > 1 else x)[: sr * 6], sr))
t = pq.read_table("data/vbd/data/test-00000-of-00001.parquet").to_pylist()
by_id = {r["id"]: r for r in t}
for sid, rate in (("p232_013", 16000), ("p257_070", 16000), ("p232_013", 8000)):
    w, s = sf.read(io.BytesIO(by_id[sid]["noisy"]["bytes"]), dtype="float32")
    clips.append((f"{sid}_{rate}", resample(w, rate), rate))

model, cfg = load_teacher("cpu")
out = {}
for name, w, rate in clips:
    with torch.no_grad():
        y = enhance(model, cfg, torch.from_numpy(np.ascontiguousarray(w))[None], rate)[0].numpy()
    out[f"{name}/x"], out[f"{name}/y"], out[f"{name}/sr"] = w, y, np.array(rate)
    print(name, rate, len(w), flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
np.savez(a.out, **out)
print("wrote", a.out)
