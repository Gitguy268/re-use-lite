"""Gate 1: RTF of teacher vs students of several depths, plus per-module time split.
usage: bench.py [--device mps|cpu] [--secs 4] [--depths 30,8,6,5,4] [--ckpt path.safetensors]"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from reuse_lite.model import load_teacher, student_from_teacher, enhance, stft_params, mag_phase_stft, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="mps"); ap.add_argument("--secs", type=float, default=4.0)
ap.add_argument("--depths", default="30,8,6,5,4"); ap.add_argument("--srs", default="8000,16000,48000")
ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()
dev = a.device


def sync():
    if dev == "mps": torch.mps.synchronize()


def timeit(f, reps):
    f(); sync(); t = time.time()
    for _ in range(reps): f()
    sync(); return (time.time() - t) / reps


teacher, cfg = load_teacher(dev)
res = {"device": dev, "secs": a.secs, "rows": []}
for sr in map(int, a.srs.split(",")):
    x = torch.randn(1, int(sr * a.secs), device=dev) * 0.1
    n_fft, hop, win = stft_params(cfg, sr)
    mag, pha, _ = mag_phase_stft(x, n_fft, hop, win, compress_factor="relu_log1p")
    for depth in map(int, a.depths.split(",")):
        m = teacher if depth == 30 else student_from_teacher(teacher, cfg, depth)[0].to(dev).eval()
        with torch.no_grad():
            full = timeit(lambda: enhance(m, cfg, x, sr), a.reps)
            # module split: run encoder once, time blocks and decoders separately
            mm = mag.transpose(1, 2).unsqueeze(1); pp = pha.transpose(1, 2).unsqueeze(1)
            inp = torch.cat((mm, pp), 1)
            B, C, T, F = inp.shape
            inp = torch.cat((inp, torch.zeros(B, C, T, 2, device=dev)), -1)
            inp = torch.cat((inp, torch.zeros(B, C, 2, F + 2, device=dev)), -2)
            enc = timeit(lambda: m.dense_encoder(inp), a.reps)
            h = m.dense_encoder(inp)
            def blocks():
                y = h
                for blk in m.TSMamba: y = blk(y)
                return y
            blk_t = timeit(blocks, a.reps)
            dec = timeit(lambda: (m.mask_decoder(h), m.phase_decoder(h)), a.reps)
        params = sum(p.numel() for p in m.parameters())
        row = dict(sr=sr, depth=depth, params=params, sec=full, rtf=full / a.secs,
                   enc=enc, blocks=blk_t, dec=dec)
        res["rows"].append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

# speedups vs teacher
for r in res["rows"]:
    t = next(q for q in res["rows"] if q["sr"] == r["sr"] and q["depth"] == 30)
    r["speedup"] = t["sec"] / r["sec"]; r["size_ratio"] = t["params"] / r["params"]
print("\nsr depth params speedup size_ratio")
for r in res["rows"]:
    print(r["sr"], r["depth"], r["params"], f"{r['speedup']:.2f}x", f"{r['size_ratio']:.2f}x")
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "results", f"bench_{dev}.json"), "w"), indent=1)
