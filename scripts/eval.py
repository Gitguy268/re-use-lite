"""Quality + speed gate: student vs teacher on the VoiceBank-DEMAND test set (824 clips).

usage: eval.py --student out/normal [--limit N] [--device mps] [--out results/gate.json]

Sets:
  vbd16    the test set as is (16 kHz), PESQ-WB
  vbd8     the same downsampled to 8 kHz, PESQ-NB
  degraded 16 kHz with extra reverb / low-pass / clipping (fixed seed), PESQ-WB
Metrics per set: PESQ, ESTOI, SI-SDR (against clean), DNSMOS OVRL/SIG/BAK (output resampled to
16 kHz), SI-SDR of the student against the teacher. Teacher scores are cached in
results/eval/teacher_<set>.json, so a second student only costs its own run.

Gate (results/gate.json "PASS"): on every set, student PESQ, ESTOI and DNSMOS OVRL/SIG/BAK are each
>= 95% of the teacher's, and the measured speed-up on vbd16 is >= 5x.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pyarrow.parquet as pq
import torch
from pesq import pesq
from pystoi import stoi
from speechmos import dnsmos

from reuse_lite.data import SRC_SR, clip, decode, lowpass, resample, reverb
from reuse_lite.model import ROOT, enhance_batch, load_student, load_teacher

ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True)
ap.add_argument("--test", default=os.path.join(ROOT, "data/vbd/data/test-00000-of-00001.parquet"))
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--device", default="mps")
ap.add_argument("--out", default=os.path.join(ROOT, "results/gate.json"))
ap.add_argument("--min-ratio", type=float, default=0.95)
ap.add_argument("--min-speedup", type=float, default=5.0)
a = ap.parse_args()
EV = os.path.join(ROOT, "results", "eval"); os.makedirs(EV, exist_ok=True)

t = pq.read_table(a.test)
ids = t.column("id").to_pylist()
n = a.limit or len(ids)
noisy16 = [decode(r["bytes"]) for r in t.column("noisy").to_pylist()[:n]]
clean16 = [decode(r["bytes"]) for r in t.column("clean").to_pylist()[:n]]
ids = ids[:n]


def degraded(i, x):
    rng = np.random.default_rng([4321, i])
    if rng.random() < 0.6:
        x, _ = reverb(x, rng)
    if rng.random() < 0.5:
        x, _ = lowpass(x, rng)
    if rng.random() < 0.4:
        x, _ = clip(x, rng)
    return x.astype(np.float32)


SETS = {
    "vbd16": (16000, noisy16, clean16),
    "vbd8": (8000, [resample(x, 8000) for x in noisy16], [resample(x, 8000) for x in clean16]),
    "degraded": (16000, [degraded(i, x) for i, x in enumerate(noisy16)], clean16),
}


def si_sdr(est, ref):
    m = min(len(est), len(ref)); est, ref = est[:m].astype(np.float64), ref[:m].astype(np.float64)
    al = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12); tt = al * ref
    return float(10 * np.log10((np.sum(tt ** 2) + 1e-12) / (np.sum((est - tt) ** 2) + 1e-12)))


def scores(est, clean, sr):
    est = np.clip(est.astype(np.float64), -1, 1)
    d16 = resample(est.astype(np.float32), 16000) if sr != 16000 else est.astype(np.float32)
    d = dnsmos.run(np.clip(d16, -1, 1), sr=16000)
    return dict(pesq=float(pesq(sr, clean, est, "wb" if sr == 16000 else "nb")),
                estoi=float(stoi(clean, est, sr, extended=True)), sisdr=si_sdr(est, clean),
                ovrl=float(d["ovrl_mos"]), sig=float(d["sig_mos"]), bak=float(d["bak_mos"]))


def run(model, cfg, sr, xs):
    outs, sec = [], 0.0
    for x in xs:
        w = torch.from_numpy(x)[None].to(a.device)
        if a.device == "mps": torch.mps.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            y = enhance_batch(model, cfg, w, sr)
        if a.device == "mps": torch.mps.synchronize()
        sec += time.time() - t0
        outs.append(y[0].float().cpu().numpy())
    return outs, sec


def mean(rows, k):
    return float(np.mean([r[k] for r in rows]))


keys = ["pesq", "estoi", "sisdr", "ovrl", "sig", "bak"]
gated = ["pesq", "estoi", "ovrl", "sig", "bak"]
report = {"student": os.path.abspath(a.student), "n_clips": n, "sets": {}}
student, scfg = load_student(a.student, a.device)
teacher = None
for name, (sr, xs, cs) in SETS.items():
    cache = os.path.join(EV, f"teacher_{name}.json")
    tc = json.load(open(cache)) if os.path.isfile(cache) else None
    if tc is None or tc["ids"] != ids:
        if teacher is None:
            teacher, tcfg = load_teacher(a.device)
        run(teacher, tcfg, sr, xs[:2])  # warm-up: first-call shader compiles don't count as speed
        print(f"{name}: teacher on {n} clips ...", flush=True)
        t_out, t_sec = run(teacher, tcfg, sr, xs)
        tc = dict(ids=ids, sec=t_sec, rows=[scores(y, c, sr) for y, c in zip(t_out, cs)],
                  noisy=[scores(x, c, sr) for x, c in zip(xs, cs)])
        np.savez_compressed(os.path.join(EV, f"teacher_{name}.npz"), *t_out)
        json.dump(tc, open(cache, "w"))
    with np.load(os.path.join(EV, f"teacher_{name}.npz")) as z:
        t_out = [z[f"arr_{i}"] for i in range(n)]
    run(student, scfg, sr, xs[:2])
    print(f"{name}: student on {n} clips ...", flush=True)
    s_out, s_sec = run(student, scfg, sr, xs)
    srows = [scores(y, c, sr) for y, c in zip(s_out, cs)]
    vs_t = float(np.mean([si_sdr(y, z) for y, z in zip(s_out, t_out)]))
    res = {"sr": sr,
           "noisy": {k: mean(tc["noisy"], k) for k in keys},
           "teacher": {k: mean(tc["rows"], k) for k in keys},
           "student": {k: mean(srows, k) for k in keys},
           "student_vs_teacher_sisdr": vs_t,
           "teacher_sec": tc["sec"], "student_sec": s_sec, "speedup": tc["sec"] / s_sec}
    res["ratio"] = {k: res["student"][k] / res["teacher"][k] for k in gated}
    res["pass"] = all(v >= a.min_ratio for v in res["ratio"].values())
    report["sets"][name] = res
    print(json.dumps({name: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["ratio"].items()},
                      "speedup": round(res["speedup"], 2), "pass": res["pass"]}), flush=True)

report["params"] = {"student": sum(p.numel() for p in student.parameters()), "teacher": 9607747}
report["speedup_vbd16"] = report["sets"]["vbd16"]["speedup"]
report["gate"] = {"quality_ok": all(s["pass"] for s in report["sets"].values()),
                  "speed_ok": report["speedup_vbd16"] >= a.min_speedup,
                  "min_ratio": a.min_ratio, "min_speedup": a.min_speedup, "full_test_set": n == len(t)}
report["gate"]["PASS"] = bool(report["gate"]["quality_ok"] and report["gate"]["speed_ok"] and report["gate"]["full_test_set"])
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(report, open(a.out, "w"), indent=1)
print("\nset       metric  noisy   teacher student ratio")
for name, s in report["sets"].items():
    for k in keys:
        r = s["student"][k] / s["teacher"][k]
        print(f"{name:9} {k:6} {s['noisy'][k]:7.3f} {s['teacher'][k]:7.3f} {s['student'][k]:7.3f} {r:6.3f}")
print(f"speed-up on vbd16: {report['speedup_vbd16']:.2f}x   GATE: {'PASS' if report['gate']['PASS'] else 'FAIL'} {report['gate']}")
