"""Write the untrained starting student (teacher encoder/decoders + evenly spaced blocks).

usage: init_student.py --depth 4 --out out/init-d4
Used to test every conversion path before training, and as the training start point."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safetensors.torch import save_file

from reuse_lite.model import load_teacher, student_config, student_from_teacher

ap = argparse.ArgumentParser()
ap.add_argument("--depth", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
t, cfg = load_teacher("cpu")
s, _, idx = student_from_teacher(t, cfg, a.depth)
os.makedirs(a.out, exist_ok=True)
save_file({k: v.contiguous() for k, v in s.state_dict().items()}, os.path.join(a.out, "model.safetensors"))
json.dump(student_config(cfg, a.depth), open(os.path.join(a.out, "config.json"), "w"), indent=2)
print(f"{a.out}: depth {a.depth}, teacher blocks {idx}, params {sum(p.numel() for p in s.parameters())}")
