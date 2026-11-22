"""Build the degraded training set from VoiceBank-DEMAND train parquet files.

usage: prep_data.py --parquet 'data/vbd/data/train-*.parquet' --out data/train --variants 2 [--limit N]
Writes <out>/sr<rate>/{noisy,clean}.f32 (raw float32, rows x samples), meta.jsonl and info.json.
Deterministic: the RNG is seeded per (clip index, variant).
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pyarrow.parquet as pq

from reuse_lite.data import RATES, decode, degrade

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--variants", type=int, default=2)
ap.add_argument("--samples", type=int, default=24000)
ap.add_argument("--limit", type=int, default=0, help="only the first N clips (smoke test)")
ap.add_argument("--val", type=int, default=0, help="hold out the last N clips into <out>_val")
ap.add_argument("--seed", type=int, default=1234)
a = ap.parse_args()

files = sorted(glob.glob(a.parquet))
assert files, a.parquet
rows = []
for f in files:
    t = pq.read_table(f, columns=["id"])
    rows += [(f, i, s) for i, s in enumerate(t.column("id").to_pylist())]
if a.limit:
    rows = rows[:a.limit]
splits = {a.out: rows[:len(rows) - a.val] if a.val else rows}
if a.val:
    splits[a.out + "_val"] = rows[len(rows) - a.val:]

for out, rws in splits.items():
    if os.path.isfile(os.path.join(out, "DONE")):
        print(out, "already built; delete it to rebuild"); continue
    os.makedirs(out, exist_ok=True)
    fh = {}
    counts = {sr: 0 for sr in RATES}

    def files(sr):
        if sr not in fh:
            od = os.path.join(out, f"sr{sr}"); os.makedirs(od, exist_ok=True)
            fh[sr] = [open(os.path.join(od, "noisy.f32"), "wb"), open(os.path.join(od, "clean.f32"), "wb"),
                      open(os.path.join(od, "meta.jsonl"), "w")]
        return fh[sr]

    cache = {}
    for k, (f, i, sid) in enumerate(rws):
        if f not in cache:
            cache.clear(); cache[f] = pq.read_table(f)
        r = cache[f].slice(i, 1).to_pylist()[0]
        assert r["id"] == sid
        noisy, clean = decode(r["noisy"]["bytes"]), decode(r["clean"]["bytes"])
        assert len(noisy) == len(clean), sid
        for v in range(a.variants):
            rng = np.random.default_rng([a.seed, k if out == a.out else 10_000_000 + k, v])
            x, c, sr, flags = degrade(noisy, clean, rng, a.samples)
            assert x.shape == c.shape == (a.samples,) and np.isfinite(x).all() and np.isfinite(c).all()
            fn, fc, fm = files(sr)
            fn.write(x.astype("<f4").tobytes()); fc.write(c.astype("<f4").tobytes())
            fm.write(json.dumps(dict(id=sid, variant=v, **flags)) + "\n")
            counts[sr] += 1
        if (k + 1) % 500 == 0:
            print(f"{out}: {k + 1}/{len(rws)} clips", flush=True)
    for sr, hs in fh.items():
        for h in hs:
            h.close()
        json.dump({"sr": sr, "rows": counts[sr], "samples": a.samples},
                  open(os.path.join(out, f"sr{sr}", "info.json"), "w"))
        print(f"{out}/sr{sr}: {counts[sr]} crops")
    open(os.path.join(out, "DONE"), "w").write("ok\n")
