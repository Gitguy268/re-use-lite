"""Teacher labelling: run the 30-block teacher on every prepared crop and store its output.

usage: label.py --data data/train [--device mps] [--batch 8]
Writes <data>/sr<rate>/teacher.f32 next to noisy.f32. Resumable: rows already written are kept
(the file length says how many), so just run it again after a stop.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from reuse_lite.data import RATES
from reuse_lite.model import enhance_batch, load_teacher

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--device", default="mps")
ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()

model, cfg = load_teacher(a.device)
total_t, total_n = 0.0, 0
for sr in RATES:
    d = os.path.join(a.data, f"sr{sr}")
    if not os.path.isfile(os.path.join(d, "info.json")):
        continue
    info = json.load(open(os.path.join(d, "info.json")))
    rows, samples = info["rows"], info["samples"]
    noisy = np.memmap(os.path.join(d, "noisy.f32"), np.float32, "r", shape=(rows, samples))
    out_path = os.path.join(d, "teacher.f32")
    row_bytes = 4 * samples
    done = os.path.getsize(out_path) // row_bytes if os.path.isfile(out_path) else 0
    with open(out_path, "r+b" if os.path.isfile(out_path) else "wb") as fh:
        fh.truncate(done * row_bytes)  # drop a half-written row after a crash
        fh.seek(done * row_bytes)
        bs = a.batch  # every crop has the same sample count, so about the same cost at every rate
        print(f"sr{sr}: {done}/{rows} already done, batch {bs}", flush=True)
        for s0 in range(done, rows, bs):
            x = torch.from_numpy(np.array(noisy[s0:s0 + bs])).to(a.device)
            t0 = time.time()
            with torch.inference_mode():
                y = enhance_batch(model, cfg, x, sr).float().cpu().numpy()
            if a.device == "mps":
                torch.mps.synchronize()
            dt = time.time() - t0
            if not np.isfinite(y).all():
                raise SystemExit(f"non-finite teacher output at sr{sr} rows {s0}..{s0 + len(y)}")
            fh.write(y.astype("<f4").tobytes()); fh.flush(); os.fsync(fh.fileno())
            total_t += dt; total_n += len(y)
            if (s0 // bs) % 25 == 0:
                left = rows - (s0 + len(y))
                print(f"sr{sr}: {s0 + len(y)}/{rows}  {dt / len(y):.3f} s/crop  ~{left * dt / len(y) / 3600:.2f} h left in this rate",
                      flush=True)
    assert os.path.getsize(out_path) == rows * row_bytes, out_path
print(f"labelled {total_n} crops in {total_t / 3600:.2f} h")
