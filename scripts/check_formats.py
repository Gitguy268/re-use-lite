"""Run the same inputs through every format and compare each against PyTorch on CPU (SI-SDR, dB).
(PyTorch on MPS itself differs from CPU by ~45 dB: float drift in the Metal scan.)

usage: check_formats.py --normal out/normal [--mlx out/mlx] [--gguf out/gguf/a.gguf ...]
                        [--browser out/check/browser] --work out/check [--n 8]
Inputs: mic_test2.wav (44.1 kHz) + VoiceBank test clips at 16 kHz, one at 8 kHz, one at 48 kHz.
Pass: f32 formats > 40 dB, reduced GGUF types (f16/q8_0 in the name) > 30 dB.
Writes <work>/report.json with "PASS".
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from reuse_lite.data import decode, resample
from reuse_lite.fsutil import rmtree
from reuse_lite.model import ROOT, VENDOR

TOOLS = os.environ.get("TOOLS", os.path.expanduser("~/re-use-lite-tools"))
ap = argparse.ArgumentParser()
ap.add_argument("--normal", required=True)
ap.add_argument("--mlx")
ap.add_argument("--gguf", nargs="*", default=[])
ap.add_argument("--browser", help="folder of browser outputs named like the inputs (from web_check.mjs)")
ap.add_argument("--work", required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--keep-inputs", action="store_true")
a = ap.parse_args()
inp = os.path.join(a.work, "inputs")


def si_sdr(e, r):
    n = min(len(e), len(r)); e, r = e[:n].astype(np.float64), r[:n].astype(np.float64)
    al = np.dot(e, r) / (np.dot(r, r) + 1e-12); t = al * r
    return float(10 * np.log10((np.sum(t ** 2) + 1e-20) / (np.sum((e - t) ** 2) + 1e-20)))


if not (a.keep_inputs and os.path.isdir(inp)):
    rmtree(inp); os.makedirs(inp)
    x, sr = sf.read(os.path.join(VENDOR, "noisy_audio", "mic_test2.wav"), dtype="float32")
    sf.write(os.path.join(inp, "mic_test2_44k.wav"), x, sr, subtype="FLOAT")
    t = pq.read_table(os.path.join(ROOT, "data/vbd/data/test-00000-of-00001.parquet"))
    rows = t.column("noisy").to_pylist()
    ids = t.column("id").to_pylist()
    step = max(1, len(rows) // a.n)
    pick = list(range(0, len(rows), step))[:a.n]
    for k, i in enumerate(pick):
        x = decode(rows[i]["bytes"])
        if k == 0:
            sf.write(os.path.join(inp, f"{ids[i]}_8k.wav"), resample(x, 8000), 8000, subtype="FLOAT")
        elif k == 1:
            sf.write(os.path.join(inp, f"{ids[i]}_48k.wav"), resample(x, 48000), 48000, subtype="FLOAT")
        else:
            sf.write(os.path.join(inp, f"{ids[i]}_16k.wav"), x, 16000, subtype="FLOAT")
names = sorted(f for f in os.listdir(inp) if f.endswith(".wav") and not f.startswith("._"))

outs = {}
d = os.path.join(a.work, "torch"); rmtree(d)
subprocess.run([sys.executable, os.path.join(os.path.abspath(a.normal), "inference.py"), inp, "-o", d,
                "--chunk-seconds", "0", "--device", "cpu"], check=True)  # CPU: deterministic reference
for f in names:  # inference.py adds _enhanced
    os.replace(os.path.join(d, f[:-4] + "_enhanced.wav"), os.path.join(d, f))
outs["torch"] = d
if a.mlx:
    d = os.path.join(a.work, "mlx"); rmtree(d)
    subprocess.run([os.path.join(TOOLS, ".venv-mlx/bin/python"), os.path.join(ROOT, "scripts/mlx_enhance.py"),
                    "--model", a.mlx, "--inp", inp, "--out", d], check=True)
    outs["mlx"] = d
cli = os.path.join(TOOLS, "ext/audio.cpp/build/cpu/bin/audiocpp_cli")
for g in a.gguf:
    tag = "gguf-" + os.path.basename(g)[:-5]
    d = os.path.join(a.work, tag); rmtree(d); os.makedirs(d)
    for f in names:
        r = subprocess.run([cli, "--task", "s2s", "--family", "reuse", "--model", g, "--backend", "cpu",
                            "--threads", str(a.threads), "--audio", os.path.join(inp, f), "--out", os.path.join(d, f)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode:
            print(r.stdout[-3000:]); raise SystemExit(f"audiocpp_cli failed on {g} {f}")
    outs[tag] = d
if a.browser:
    outs["browser"] = a.browser

ref = {f: sf.read(os.path.join(outs["torch"], f), dtype="float32")[0] for f in names}
rep = {"inputs": names, "formats": {}}
ok = True
for tag, d in outs.items():
    if tag == "torch":
        continue
    per = {}
    for f in names:
        y, _ = sf.read(os.path.join(d, f), dtype="float32")
        y = y.mean(1) if y.ndim > 1 else y
        assert len(y) == len(ref[f]), (tag, f, len(y), len(ref[f]))
        assert np.isfinite(y).all(), (tag, f)
        per[f] = si_sdr(y, ref[f])
    need = 30.0 if ("f16" in tag or "q8" in tag or "bf16" in tag) else 40.0
    worst = min(per.values())
    rep["formats"][tag] = {"min_db": worst, "mean_db": float(np.mean(list(per.values()))), "need_db": need,
                           "pass": worst > need, "per_file": per}
    ok &= worst > need
    print(f"{tag:28} min {worst:6.1f} dB  mean {np.mean(list(per.values())):6.1f} dB  need > {need:.0f}  "
          f"{'PASS' if worst > need else 'FAIL'}")
rep["PASS"] = bool(ok)
json.dump(rep, open(os.path.join(a.work, "report.json"), "w"), indent=1)
print("CHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
