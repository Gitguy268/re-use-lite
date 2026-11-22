"""Assemble the normal (PyTorch) release folder from a trained student.

usage: export.py --student ckpt/run1/best --out out/normal
Copies weights + config, NVIDIA's model code (unchanged), the pure-PyTorch Mamba, inference.py,
LICENSE (OpenMDW-1.1 with NVIDIA's notice) and NOTICE, then makes before/after samples with the
bundled inference.py, run as a separate process from the release folder itself.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from reuse_lite.data import decode
from reuse_lite.fsutil import rmtree
from reuse_lite.model import ROOT, VENDOR

ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--no-samples", action="store_true")
a = ap.parse_args()
a.out = os.path.abspath(a.out)

rmtree(a.out)
os.makedirs(os.path.join(a.out, "models")); os.makedirs(os.path.join(a.out, "reuse_lite"))
cfg = json.load(open(os.path.join(a.student, "config.json")))
assert cfg.get("model_type") == "nemotron_reuse" and 1 <= cfg["model_cfg"]["num_tfmamba"] <= 30, cfg
for f in ("config.json", "model.safetensors"):
    shutil.copyfile(os.path.join(a.student, f), os.path.join(a.out, f))
for f in ("generator_SEMamba_time_d4.py", "codec_module_time_d4.py", "mamba_block2_SEMamba.py", "stfts.py"):
    shutil.copyfile(os.path.join(VENDOR, "models", f), os.path.join(a.out, "models", f))
for f in ("__init__.py", "mamba_torch.py", "scan_metal.py"):
    shutil.copyfile(os.path.join(ROOT, "reuse_lite", f), os.path.join(a.out, "reuse_lite", f))
for f in ("inference.py", "requirements.txt", "NOTICE"):
    shutil.copyfile(os.path.join(ROOT, "release", f), os.path.join(a.out, f))
shutil.copyfile(os.path.join(VENDOR, "LICENSE"), os.path.join(a.out, "LICENSE"))

if not a.no_samples:
    sd = os.path.join(a.out, "samples"); os.makedirs(sd)
    shutil.copyfile(os.path.join(VENDOR, "noisy_audio", "mic_test2.wav"), os.path.join(sd, "mic_test2.wav"))
    t = pq.read_table(os.path.join(ROOT, "data/vbd/data/test-00000-of-00001.parquet"))
    ids = t.column("id").to_pylist()
    for sid in ("p232_013", "p257_070"):
        i = ids.index(sid)
        x = decode(t.column("noisy")[i].as_py()["bytes"])
        sf.write(os.path.join(sd, f"vbd_{sid}.wav"), x, 16000)
    subprocess.run([sys.executable, os.path.join(a.out, "inference.py"), sd, "-o", sd], check=True, cwd=a.out)
    for f in sorted(f for f in os.listdir(sd) if not f.startswith("._")):
        y, sr = sf.read(os.path.join(sd, f))
        assert np.isfinite(y).all() and np.abs(y).max() > 1e-4, f
print("exported", a.out, sorted(f for f in os.listdir(a.out) if not f.startswith("._")))
