# SPDX-License-Identifier: Apache-2.0
"""Package a RE-USE (or RE-USE Lite student) safetensors folder as an audio.cpp `reuse` GGUF.

Same procedure as audio.cpp tests/reuse/convert_reuse_gguf.py (Copyright 2026 ShugoAI LLC,
Apache-2.0), without its lock to NVIDIA's exact checkpoint hash, so a student can be packaged:
1-D tensors, A_log and the depthwise conv1d weights stay F32; everything else takes --type.
Checks afterwards: tensor inventory, shapes, and byte-exact F32 tensors.

usage: convert_gguf.py --source out/normal --output out/gguf/x-f32.gguf --audiocpp DIR [--type f32|f16|q8_0]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--source", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--audiocpp", type=Path, required=True, help="audio.cpp checkout with build/cpu/bin/audiocpp_gguf")
ap.add_argument("--type", choices=("f32", "f16", "bf16", "q8_0"), default="f32")
a = ap.parse_args()

ckpt = a.source / "model.safetensors"
cfg = json.load(open(a.source / "config.json"))
nblk = cfg["model_cfg"]["num_tfmamba"]
assert 1 <= nblk <= 30, nblk
tool = a.audiocpp / "build/cpu/bin/audiocpp_gguf"
spec = a.audiocpp / "model_specs/reuse.json"
assert tool.is_file() and spec.is_file(), (tool, spec)
a.output.parent.mkdir(parents=True, exist_ok=True)
# audio.cpp embeds every file under --root; stage only config.json (as the official package does)
stage = Path(tempfile.mkdtemp()) / a.output.stem
stage.mkdir()
shutil.copyfile(a.source / "config.json", stage / "config.json")
cmd = [str(tool.resolve()), "--input", f"weights={ckpt.resolve()}", "--output", str(a.output.resolve()),
       "--type", a.type, "--family", "reuse", "--model-spec", str(spec.resolve()),
       "--root", str(stage), "--overwrite"]
with safe_open(ckpt, framework="np") as src:
    names = list(src.keys())
    blocks = {int(k.split(".")[1]) for k in names if k.startswith("TSMamba.")}
    assert blocks == set(range(nblk)), f"config says {nblk} blocks, weights have {sorted(blocks)}"
    for name in names:
        shape = src.get_slice(name).get_shape()
        if len(shape) == 1 or name.endswith(".A_log") or name.endswith(".conv1d.weight"):
            cmd += ["--keep-type", f"weights/{name}=f32"]
r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
shutil.rmtree(stage.parent)
if r.returncode:
    print(r.stdout)
    raise SystemExit(f"audiocpp_gguf failed ({r.returncode})")

reader = gguf.GGUFReader(str(a.output))
with safe_open(ckpt, framework="np") as src:
    if {t.name for t in reader.tensors} != {"weights/" + k for k in src.keys()}:
        raise SystemExit("GGUF tensor inventory mismatch")
    ranks = reader.get_field("audiocpp.tensor_ranks").contents()
    shapes = reader.get_field("audiocpp.tensor_shapes").contents()
    cur = 0
    for t, rank in zip(reader.tensors, ranks, strict=True):
        orig = src.get_tensor(t.name.removeprefix("weights/"))
        if tuple(shapes[cur:cur + rank]) != orig.shape:
            raise SystemExit(f"GGUF shape mismatch: {t.name}")
        cur += rank
        if a.type == "f32" and t.tensor_type != gguf.GGMLQuantizationType.F32:
            raise SystemExit(f"GGUF type mismatch: {t.name}")
        if t.tensor_type == gguf.GGMLQuantizationType.F32 and not np.array_equal(
                t.data.reshape(-1).view(np.uint8), orig.reshape(-1).view(np.uint8)):
            raise SystemExit(f"F32 tensor bytes differ: {t.name}")
    if cur != len(shapes):
        raise SystemExit("trailing GGUF tensor shape metadata")
    if reader.get_field("audiocpp.embedded_files.names").contents() != ["config.json"]:
        raise SystemExit("unexpected embedded files")
sha = hashlib.sha256(a.output.read_bytes()).hexdigest()
print(f"PASS {a.output.name}: {len(reader.tensors)} tensors, {nblk} blocks, type {a.type}, "
      f"{a.output.stat().st_size / 2**20:.1f} MiB, sha256 {sha}")
