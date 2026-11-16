"""Load the upstream SEMamba (teacher) or a shallower student, on CUDA/MPS/CPU."""
import copy
import json
import os
import sys
import types

import torch
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "vendor")


def _install_mamba_shim(force_torch=False):
    if not force_torch:
        try:
            import mamba_ssm  # noqa: F401  (real CUDA kernels)
            return "mamba_ssm"
        except Exception:
            pass
    from reuse_lite.mamba_torch import Mamba
    sys.modules["mamba_ssm"] = types.SimpleNamespace(Mamba=Mamba)
    return "torch"


BACKEND = _install_mamba_shim(force_torch=os.environ.get("REUSE_TORCH_MAMBA") == "1")
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)
from models.generator_SEMamba_time_d4 import SEMamba  # noqa: E402
from models.stfts import mag_phase_stft, mag_phase_istft  # noqa: E402,F401
from utils.util import pad_or_trim_to_match  # noqa: E402,F401


def load_cfg(path=os.path.join(VENDOR, "config.json")):
    with open(path) as f:
        return json.load(f)


def load_teacher(device="cpu"):
    cfg = load_cfg()
    m = SEMamba(cfg)
    m.load_state_dict(load_file(os.path.join(VENDOR, "model.safetensors")), strict=True)
    return m.to(device).eval(), cfg


def student_from_teacher(teacher, cfg, n_blocks):
    """Copy encoder/decoders, take n_blocks evenly spaced TFMamba blocks."""
    scfg = copy.deepcopy(cfg)
    scfg["model_cfg"]["num_tfmamba"] = n_blocks
    s = SEMamba(scfg)
    nt = len(teacher.TSMamba)
    idx = [round(i * (nt - 1) / max(n_blocks - 1, 1)) for i in range(n_blocks)]
    sd = {}
    for k, v in teacher.state_dict().items():
        if k.startswith("TSMamba."):
            continue
        sd[k] = v
    for j, i in enumerate(idx):
        for k, v in teacher.TSMamba[i].state_dict().items():
            sd[f"TSMamba.{j}.{k}"] = v
    s.load_state_dict(sd, strict=True)
    return s, scfg, idx


def make_even(v):
    v = int(round(v))
    return v if v % 2 == 0 else v + 1


def stft_params(cfg, sr):
    st = cfg["stft_cfg"]
    base = st["sampling_rate"]
    return tuple(make_even(st[k] * sr // base) for k in ("n_fft", "hop_size", "win_size"))


@torch.no_grad()
def enhance(model, cfg, wav, sr):
    """wav: (1, n) float tensor on model's device. Mirrors upstream inference.py."""
    n_fft, hop, win = stft_params(cfg, sr)
    cf = cfg["model_cfg"]["compress_factor"]
    mag, pha, _ = mag_phase_stft(wav, n_fft, hop, win, compress_factor=cf, center=True, addeps=False)
    amp_g, pha_g, _ = model(mag, pha)
    m = torch.expm1(torch.relu(amp_g))
    zero_portion = torch.sum(m == 0, 1) / m.shape[1]
    amp_g[:, :, (zero_portion > 0.5)[0]] = 0
    out = mag_phase_istft(amp_g, pha_g, n_fft, hop, win, cf)
    return pad_or_trim_to_match(wav, out, pad_value=1e-8)
