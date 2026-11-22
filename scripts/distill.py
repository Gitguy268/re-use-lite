"""Distil the 30-block RE-USE teacher into a shallow student, on Apple GPUs (MPS).

usage: distill.py --data data/train --val data/train_val --out ckpt/run1 --depth 4 --steps 30000
                  [--batch 4] [--lr 2e-4] [--resume] [--max-steps N]

Targets are the teacher outputs stored by label.py (teacher.f32), plus 0.3x the VoiceBank clean
speech on crops where the clean signal is a valid target (rates up to 16 kHz).
Loss per target, in the style of SEMamba training: 0.9 magnitude MSE + 0.3 anti-wrapped phase
(instantaneous, group delay, instantaneous frequency) + 0.1 complex MSE (x2) + 0.2 time L1 +
0.1 STFT consistency (x2).

Checkpoints: <out>/step_<n>.pt (model + optimizer + scheduler + RNG), last 3 kept, latest.txt, and
<out>/best/ (config.json + model.safetensors, lowest validation loss). --resume continues from latest.
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint

from reuse_lite.data import Buckets
from reuse_lite.model import (load_teacher, mag_phase_stft, stft_params, student_config,
                              student_from_teacher, enhance_batch)

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--val", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--steps", type=int, default=30000, help="schedule length")
ap.add_argument("--max-steps", type=int, default=0, help="stop after this step (smoke test); 0 = --steps")
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--clean-weight", type=float, default=0.3)
ap.add_argument("--save-every", type=int, default=500)
ap.add_argument("--val-every", type=int, default=1000)
ap.add_argument("--val-items", type=int, default=64)
ap.add_argument("--no-grad-ckpt", action="store_true", help="keep all block activations (more memory)")
ap.add_argument("--device", default="mps")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()
dev = a.device
os.makedirs(a.out, exist_ok=True)
LOG = open(os.path.join(a.out, "train.log"), "a")


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


# ---------------- data ----------------
train = Buckets(a.data)
for sr, b in train.b.items():
    assert "teacher" in b, f"{a.data}/sr{sr}: teacher.f32 missing or incomplete; run label.py first"
sizes = train.sizes()
rates = sorted(sizes)
weights = np.array([sizes[r] for r in rates], np.float64); weights /= weights.sum()
val = Buckets(a.val) if a.val and os.path.isdir(a.val) else None
if val is not None:
    for sr, b in val.b.items():
        assert "teacher" in b, f"{a.val}/sr{sr}: teacher.f32 missing; run label.py on it"
log(f"train crops per rate: {sizes}; val: {val.sizes() if val else None}")


def batch_from(bk, sr, idx):
    b = bk.b[sr]
    x = torch.from_numpy(np.stack([b["noisy"][i] for i in idx])).to(dev)
    t = torch.from_numpy(np.stack([b["teacher"][i] for i in idx])).to(dev)
    c = torch.from_numpy(np.stack([b["clean"][i] for i in idx])).to(dev)
    ok = torch.tensor([bool(b["meta"][i].get("clean_ok")) for i in idx], device=dev)
    return x, t, c, ok


# ---------------- model ----------------
teacher, cfg = load_teacher("cpu")
student, _, picked = student_from_teacher(teacher, cfg, a.depth)
del teacher
scfg = student_config(cfg, a.depth)
student = student.to(dev).train()
cf = cfg["model_cfg"]["compress_factor"]
if not a.no_grad_ckpt:
    # recompute each TFMamba block in backward: the scan state for one block is hundreds of MB
    for blk in student.TSMamba:
        fwd = blk.forward
        blk.forward = (lambda f: (lambda x: checkpoint(f, x, use_reentrant=False)))(fwd)
log(f"student depth {a.depth} from teacher blocks {picked}; params {sum(p.numel() for p in student.parameters())}")

opt = torch.optim.AdamW(student.parameters(), lr=a.lr, betas=(0.8, 0.99), weight_decay=0.01)
warm = min(500, a.steps // 20)


def lr_at(step):
    if step < warm:
        return a.lr * (step + 1) / warm
    p = (step - warm) / max(1, a.steps - warm)
    return 1e-6 + 0.5 * (a.lr - 1e-6) * (1 + math.cos(math.pi * min(1.0, p)))


# ---------------- loss ----------------
def anti_wrap(x):
    return torch.abs(x - torch.round(x / (2 * math.pi)) * 2 * math.pi)


def se_loss(out, amp_g, pha_g, com_g, ref, sr, w=None):
    """Per-item loss vector (B,) against a reference waveform ref (B, n)."""
    n_fft, hop, win = stft_params(cfg, sr)
    mag_r, pha_r, com_r = mag_phase_stft(ref, n_fft, hop, win, compress_factor=cf, center=True, addeps=True)
    T = min(mag_r.shape[-1], amp_g.shape[-1])
    mag_r, pha_r, com_r = mag_r[..., :T], pha_r[..., :T], com_r[:, :, :T]
    amp, pha, com = amp_g[..., :T], pha_g[..., :T], com_g[:, :, :T]
    l_mag = ((amp - mag_r) ** 2).mean((1, 2))
    ip = anti_wrap(pha_r - pha).mean((1, 2))
    gd = anti_wrap(torch.diff(pha_r, dim=1) - torch.diff(pha, dim=1)).mean((1, 2))
    iaf = anti_wrap(torch.diff(pha_r, dim=2) - torch.diff(pha, dim=2)).mean((1, 2))
    l_pha = ip + gd + iaf
    l_com = ((com - com_r) ** 2).mean((1, 2, 3)) * 2
    n = min(out.shape[-1], ref.shape[-1])
    l_time = (out[:, :n] - ref[:, :n]).abs().mean(1)
    _, _, com_o = mag_phase_stft(out, n_fft, hop, win, compress_factor=cf, center=True, addeps=True)
    To = min(com_o.shape[2], com.shape[2])
    l_con = ((com_o[:, :, :To] - com[:, :, :To]) ** 2).mean((1, 2, 3)) * 2
    tot = 0.9 * l_mag + 0.3 * l_pha + 0.1 * l_com + 0.2 * l_time + 0.1 * l_con
    parts = dict(mag=l_mag.mean().item(), pha=l_pha.mean().item(), com=l_com.mean().item(),
                 time=l_time.mean().item(), con=l_con.mean().item())
    return tot, parts


def step_loss(x, t, c, ok, sr):
    out, amp_g, pha_g, com_g = enhance_batch(student, cfg, x, sr, raw=True)
    lt, parts = se_loss(out, amp_g, pha_g, com_g, t, sr)
    loss = lt.mean()
    if a.clean_weight > 0 and ok.any():
        lc, _ = se_loss(out[ok], amp_g[ok], pha_g[ok], com_g[ok], c[ok], sr)
        loss = loss + a.clean_weight * lc.sum() / len(ok)
    return loss, parts


@torch.no_grad()
def validate():
    if val is None:
        return None
    student.eval()
    tot, n = 0.0, 0
    for sr, b in sorted(val.b.items()):
        m = min(len(b["meta"]), max(1, a.val_items * len(b["meta"]) // max(1, sum(val.sizes().values()))))
        for s0 in range(0, m, a.batch):
            idx = list(range(s0, min(m, s0 + a.batch)))
            x, t, _, _ = batch_from(val, sr, idx)
            out, amp_g, pha_g, com_g = enhance_batch(student, cfg, x, sr, raw=True)
            lt, _ = se_loss(out, amp_g, pha_g, com_g, t, sr)
            tot += lt.sum().item(); n += len(idx)
    student.train()
    return tot / max(1, n)


def save_folder(path):
    os.makedirs(path, exist_ok=True)
    sd = {k: v.detach().float().cpu().contiguous() for k, v in student.state_dict().items()}
    save_file(sd, os.path.join(path + "/model.safetensors"))
    json.dump(scfg, open(os.path.join(path, "config.json"), "w"), indent=2)


def save_ckpt(step, best):
    p = os.path.join(a.out, f"step_{step:07d}.pt")
    torch.save(dict(step=step, model=student.state_dict(), opt=opt.state_dict(), best=best,
                    depth=a.depth, picked=picked, args=vars(a),
                    rng=dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state())),
               p + ".tmp")
    os.replace(p + ".tmp", p)
    open(os.path.join(a.out, "latest.txt"), "w").write(os.path.basename(p) + "\n")
    for old in sorted(glob.glob(os.path.join(a.out, "step_*.pt")))[:-3]:
        os.remove(old)


# ---------------- resume ----------------
step, best = 0, float("inf")
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
latest = os.path.join(a.out, "latest.txt")
if a.resume and os.path.isfile(latest):
    ck = torch.load(os.path.join(a.out, open(latest).read().strip()), map_location="cpu", weights_only=False)
    assert ck["depth"] == a.depth, f"checkpoint depth {ck['depth']} != --depth {a.depth}"
    student.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    step, best = ck["step"], ck["best"]
    random.setstate(ck["rng"]["py"]); np.random.set_state(ck["rng"]["np"]); torch.set_rng_state(ck["rng"]["torch"])
    log(f"resumed from step {step} (best val {best:.5f})")
elif a.resume:
    log("--resume given but no checkpoint yet; starting from step 0")

# ---------------- train ----------------
stop = a.max_steps or a.steps
t_last, n_last = time.time(), step
while step < stop:
    sr = int(np.random.choice(rates, p=weights))
    idx = np.random.randint(0, sizes[sr], a.batch).tolist()
    x, t, c, ok = batch_from(train, sr, idx)
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    loss, parts = step_loss(x, t, c, ok, sr)
    if not torch.isfinite(loss):
        raise SystemExit(f"non-finite loss at step {step}; restart with --resume (it reloads the last good checkpoint)")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
    opt.step()
    step += 1
    if step % 50 == 0 or step == stop or step <= 3:
        if dev == "mps":
            torch.mps.synchronize()
        dt = (time.time() - t_last) / max(1, step - n_last); t_last, n_last = time.time(), step
        mem = torch.mps.driver_allocated_memory() / 2**30 if dev == "mps" else 0
        log(f"step {step}/{stop} sr {sr} loss {loss.item():.4f} " +
            " ".join(f"{k} {v:.4f}" for k, v in parts.items()) +
            f" gnorm {gn.item():.2f} lr {lr_at(step):.2e} {dt:.2f}s/step mem {mem:.1f}GB "
            f"eta {(stop - step) * dt / 3600:.1f}h")
    if step % a.val_every == 0 or step == stop:
        v = validate()
        if v is not None:
            log(f"val {v:.5f} (best {best:.5f})")
            if v < best:
                best = v; save_folder(os.path.join(a.out, "best")); log("saved best/")
    if step % a.save_every == 0 or step == stop:
        save_ckpt(step, best)
save_folder(os.path.join(a.out, "last"))
if val is None:
    save_folder(os.path.join(a.out, "best"))
log(f"done at step {step}")
