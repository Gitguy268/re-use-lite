# SPDX-License-Identifier: Apache-2.0
"""Run the GPU stages (data prep, teacher labelling, distillation) on Modal instead of the Mac.

usage (through run.sh, which supplies modal==1.6.4 via uv):
  bash run.sh modal probe                 # ~5 min: CUDA parity vs CPU reference + speed per GPU
  bash run.sh modal prep                  # CPU: download VoiceBank-DEMAND, build data/train(_val) on the volume
  bash run.sh modal label [--hours H]     # GPU: teacher labels every crop
  bash run.sh modal train [--hours H]     # GPU: distillation, detached; resumable with the same command
  bash run.sh modal status                # last log lines + call state
  bash run.sh modal download              # best/, last/, train.log -> ckpt/run-d$DEPTH

Every launch checks the Modal credit: it refuses unless (credit left this month - $1 - the possibly
unbilled last 1.5 h of earlier launches) covers this launch's worst case
(GPU $/h x timeout). The timeout is the hard stop, so a hang can't spend more than that.
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
APP = "re-use-lite"
VOL = "re-use-lite-data"
GPU = os.environ.get("MODAL_GPU", "L4")  # applied per call with .with_options(gpu=...)
TF32 = os.environ.get("REUSE_TF32") == "1"
DEPTH = int(os.environ.get("DEPTH", "4"))
STEPS = int(os.environ.get("STEPS", "30000"))
BATCH = int(os.environ.get("BATCH", "2"))
VBD_REPO, VBD_REV = "JacobLinCool/VoiceBank-DEMAND-16k", "4497db342d7312978c45690591fda86117831940"
EXPECTED_ROWS = {"train": {8000: 2165, 16000: 13993, 24000: 4093, 48000: 2693},
                 "train_val": {8000: 11, 16000: 127, 24000: 38, 48000: 24}}
MONTHLY_CREDIT = float(os.environ.get("MODAL_MONTHLY_CREDIT", "30"))
CREDIT_FLOOR = 1.0
# Conservative $/hour per container, GPU + the CPU/memory we request (rounded up from Modal's list prices).
USD_PER_H = {"L4": 0.95, "A10G": 1.25, "L40S": 2.10, "A100-40GB": 2.80, "H100": 4.50, "cpu": 0.30}
LEDGER = Path.home() / ".local/state/re-use-lite-modal/ledger.json"

MAMBA_WHL = ("https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/"
             "mamba_ssm-2.3.2.post1%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
             "mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
             "7a67070c1e7e99c95abd1319623f044e8a1b3fb46f774bfdea949f0a4fc79638")
CCONV_WHL = ("https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/"
             "causal_conv1d-1.7.0%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
             "causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
             "8e81f8c76435ad31aa41edc6c0c9d26de971a293b190183d2816da71547614d7")


def _whl(url, name, sha):
    return (f"curl -fsSL -o '/tmp/{name}' '{url}' && echo '{sha}  /tmp/{name}' | sha256sum -c - && "
            f"pip install --no-deps '/tmp/{name}' && rm '/tmp/{name}'")


skip = ["**/__pycache__", "**/._*", "**/.DS_Store"]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    # same numerics stack as requirements/torch.lock.txt; transformers only because mamba_ssm imports it
    .pip_install("numpy==2.5.5", "scipy==1.18.1", "soundfile==0.14.0", "pyarrow==26.0.1", "safetensors==0.8.0",
                 "einops==0.8.2", "pyyaml==6.0.3", "packaging", "transformers==4.46.3")
    .run_commands(_whl(*MAMBA_WHL), _whl(*CCONV_WHL))
    .env({"REUSE_REQUIRE_MAMBA_SSM": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "reuse_lite", "/app/reuse_lite", ignore=skip)
    .add_local_dir(ROOT / "scripts", "/app/scripts", ignore=skip + ["*.mjs", "*.sh"])
    .add_local_dir(ROOT / "vendor/models", "/app/vendor/models", ignore=skip)
    .add_local_dir(ROOT / "vendor/utils", "/app/vendor/utils", ignore=skip)
    .add_local_file(ROOT / "vendor/config.json", "/app/vendor/config.json")
    .add_local_file(ROOT / "vendor/model.safetensors", "/app/vendor/model.safetensors")
)
app = modal.App(APP)
vol = modal.Volume.from_name(VOL, create_if_missing=True)
V = "/vol"


def _run(cmd, log=None):
    """Run a command in /app, stream its output (and append it to log)."""
    print("+", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd="/app", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    fh = open(log, "a") if log else None
    for line in p.stdout:
        print(line, end="", flush=True)
        if fh:
            fh.write(line); fh.flush()
    if p.wait():
        raise RuntimeError(f"{cmd[1:3]} failed with exit code {p.returncode}")


def _copy_data(names, dst="/tmp/data", with_teacher=True):
    """Volume -> local disk: random memmap reads over the volume mount are slow."""
    import shutil
    for n in names:
        for sr_dir in sorted(Path(V, "data", n).glob("sr*")):
            out = Path(dst, n, sr_dir.name); out.mkdir(parents=True, exist_ok=True)
            for f in ("noisy.f32", "clean.f32", "meta.jsonl", "info.json") + (("teacher.f32",) if with_teacher else ()):
                if (sr_dir / f).exists():
                    shutil.copyfile(sr_dir / f, out / f)
        Path(dst, n, "DONE").write_text("ok\n")


@app.function(image=image, cpu=4, memory=8192, timeout=5400, volumes={V: vol})
def prep():
    from huggingface_hub import snapshot_download
    snapshot_download(VBD_REPO, repo_type="dataset", revision=VBD_REV, allow_patterns=["data/train-*"],
                      local_dir="/tmp/vbd")
    _run([sys.executable, "scripts/prep_data.py", "--parquet", "/tmp/vbd/data/train-*.parquet",
          "--out", "/tmp/data/train", "--variants", "2", "--val", "100"])
    for n, want in EXPECTED_ROWS.items():
        got = {int(p.name[2:]): json.loads((p / "info.json").read_text())["rows"]
               for p in Path("/tmp/data", n).glob("sr*")}
        if got != want:
            raise RuntimeError(f"{n}: rows {got} differ from the Mac build {want}")
    import hashlib
    import shutil
    if Path(V, "data").exists():
        shutil.rmtree(Path(V, "data"))
    shutil.copytree("/tmp/data", Path(V, "data"))
    vol.commit()
    return {n: hashlib.sha256(Path("/tmp/data", n, "sr16000/meta.jsonl").read_bytes()).hexdigest()
            for n in EXPECTED_ROWS}


def _sisdr(ref, est):
    import numpy as np
    n = min(len(ref), len(est)); ref, est = ref[:n].astype(np.float64), est[:n].astype(np.float64)
    a = (est @ ref) / (ref @ ref); e = est - a * ref
    return float(10 * np.log10((a * a * (ref @ ref)) / max(e @ e, 1e-30)))


@app.function(image=image, gpu="L4", timeout=1200)
def probe(ref_npz: bytes, tf32: bool = False, label_batch: int = 16):
    os.environ["REUSE_TF32"] = "1" if tf32 else "0"
    import numpy as np
    import torch
    sys.path.insert(0, "/app")
    from reuse_lite import model as M
    from reuse_lite.data import RATES
    assert M.BACKEND == "mamba_ssm", M.BACKEND
    res = {"gpu": torch.cuda.get_device_name(0), "tf32": tf32, "label_batch": label_batch, "parity_db": {}, "label_s_per_crop": {}, "train_s_per_step": {}}
    teacher, cfg = M.load_teacher("cuda")
    ref = np.load(io.BytesIO(ref_npz))
    for n in sorted({k.split("/")[0] for k in ref.files}):
        x = torch.from_numpy(np.ascontiguousarray(ref[f"{n}/x"]))[None].cuda()
        y = M.enhance(teacher, cfg, x, int(ref[f"{n}/sr"]))[0].cpu().numpy()
        res["parity_db"][n] = round(_sisdr(ref[f"{n}/y"], y), 1)
    print(res, flush=True)
    for sr in RATES:
        x = torch.randn(label_batch, 24000, device="cuda") * 0.1
        with torch.inference_mode():
            M.enhance_batch(teacher, cfg, x, sr); torch.cuda.synchronize(); t0 = time.time()
            for _ in range(3):
                M.enhance_batch(teacher, cfg, x, sr)
            torch.cuda.synchronize()
        res["label_s_per_crop"][sr] = round((time.time() - t0) / (3 * label_batch), 4)
    print(res, flush=True)
    del teacher; torch.cuda.empty_cache()
    # synthetic buckets: only the shapes matter for timing
    rng = np.random.default_rng(0)
    for sr in RATES:
        d = Path(f"/tmp/probe/sr{sr}"); d.mkdir(parents=True, exist_ok=True)
        for f in ("noisy", "clean", "teacher"):
            (rng.standard_normal((64, 24000)).astype("<f4") * 0.1).tofile(d / f"{f}.f32")
        (d / "meta.jsonl").write_text("".join(json.dumps({"id": i, "clean_ok": sr <= 16000}) + "\n" for i in range(64)))
        (d / "info.json").write_text(json.dumps({"sr": sr, "rows": 64, "samples": 24000}))
    Path("/tmp/probe/DONE").write_text("ok\n")
    for b, extra in ((2, []), (2, ["--no-grad-ckpt"]), (8, ["--no-grad-ckpt"])):
        key = f"batch{b}" + ("_nockpt" if extra else "")
        out = subprocess.run([sys.executable, "scripts/distill.py", "--data", "/tmp/probe", "--out", f"/tmp/p_{key}",
                              "--depth", str(DEPTH), "--steps", "30000", "--max-steps", "100", "--batch", str(b),
                              "--device", "cuda", "--val-every", "100000", "--save-every", "100000"] + extra,
                             cwd="/app", capture_output=True, text=True)
        m = re.findall(r"step 100/100 .* ([\d.]+)s/step", out.stdout)
        res["train_s_per_step"][key] = float(m[-1]) if m else (out.stdout + out.stderr)[-800:]
        print(key, res["train_s_per_step"][key], flush=True)
    return res


@app.function(image=image, gpu="L4", timeout=7200, volumes={V: vol})
def label(batch: int = 16, tf32: bool = False):
    os.environ["REUSE_TF32"] = "1" if tf32 else "0"
    import shutil
    _copy_data(["train_val", "train"], with_teacher=False)
    log = f"{V}/logs/label.log"; Path(log).parent.mkdir(parents=True, exist_ok=True)
    for n in ("train_val", "train"):
        _run([sys.executable, "scripts/label.py", "--data", f"/tmp/data/{n}", "--device", "cuda",
              "--batch", str(batch)], log)
        for t in Path("/tmp/data", n).glob("sr*/teacher.f32"):
            shutil.copyfile(t, Path(V, "data", n, t.parent.name, "teacher.f32"))
        vol.commit()
    return "labelled"


@app.function(image=image, gpu="L4", timeout=86400, volumes={V: vol})
def train(depth: int, steps: int, batch: int, extra: list, tf32: bool = False):
    os.environ["REUSE_TF32"] = "1" if tf32 else "0"
    import threading
    _copy_data(["train", "train_val"])
    out = f"{V}/ckpt/run-d{depth}"
    Path(out).mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def committer():  # checkpoints (every 500 steps) reach the volume even if the container dies
        while not stop.wait(300):
            vol.commit()
    threading.Thread(target=committer, daemon=True).start()
    try:
        _run([sys.executable, "scripts/distill.py", "--data", "/tmp/data/train", "--val", "/tmp/data/train_val",
              "--out", out, "--depth", str(depth), "--steps", str(steps), "--batch", str(batch), "--device", "cuda",
              "--resume"] + list(extra))
    finally:
        stop.set(); vol.commit()
    return open(f"{out}/train.log").read()[-2000:]


# ---------------- local side ----------------
def credit_left():
    out = subprocess.run([sys.executable, "-m", "modal", "billing", "summary", "--json"],
                         capture_output=True, text=True, check=True).stdout
    return MONTHLY_CREDIT + float(json.loads(out)["adjustments"]["credits"])


BILLING_LAG_H = 1.5


def _ledger():
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else []


def budget(kind, hours, what):
    """Refuse unless credit left - $1 - possibly-unbilled usage covers this launch's worst case.

    Possibly unbilled = each earlier launch's worst-case $/h over the part of its run (start to end,
    or to its timeout if still running / unknown) that falls in the last BILLING_LAG_H hours."""
    worst = round(USD_PER_H[kind] * hours, 2)
    rows, now = _ledger(), time.time()
    unbilled = 0.0
    for r in rows:
        end = r.get("end") or r["t"] + r["hours"] * 3600
        overlap = max(0.0, min(end, now) - max(r["t"], now - BILLING_LAG_H * 3600)) / 3600
        unbilled += USD_PER_H[r["gpu"]] * overlap
    left = credit_left()
    print(f"Modal credit left ${left:.2f}; possibly not yet billed ${unbilled:.2f}; "
          f"this launch worst case ${worst:.2f} ({kind}, {hours:g} h)")
    if left - CREDIT_FLOOR - unbilled < worst:
        raise SystemExit("Refusing: not enough Modal credit for the worst case (rule: keep $1 spare). "
                         "Use a smaller --hours, a cheaper GPU, or wait for billing / the monthly credit.")
    rows.append({"t": now, "what": what, "worst_usd": worst, "gpu": kind, "hours": hours})
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(rows, indent=1))
    return len(rows) - 1


def finished(i):
    rows = _ledger(); rows[i]["end"] = time.time(); LEDGER.write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["probe", "prep", "label", "train", "status", "download"])
    ap.add_argument("--hours", type=float, default=None, help="hard timeout (and budget) for label/train")
    ap.add_argument("--extra", default="", help="extra distill.py flags, e.g. '--no-grad-ckpt'")
    a = ap.parse_args()
    state = LEDGER.parent / f"train-d{DEPTH}.json"
    if a.stage == "status":
        try:
            print(b"".join(vol.read_file(f"ckpt/run-d{DEPTH}/train.log")).decode()[-1500:])
        except Exception as e:  # noqa: BLE001
            print("no training log on the volume yet:", e)
        if state.exists():
            cid = json.loads(state.read_text())["call"]
            try:
                modal.FunctionCall.from_id(cid).get(timeout=0); print("train call: finished")
            except TimeoutError:
                print("train call: running")
            except Exception as e:  # noqa: BLE001
                print("train call:", type(e).__name__, e)
        return
    if a.stage == "download":
        dst = ROOT / f"ckpt/run-d{DEPTH}"; dst.mkdir(parents=True, exist_ok=True)
        for p in ("best", "last", "train.log", "latest.txt"):
            subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", VOL, f"ckpt/run-d{DEPTH}/{p}",
                            str(dst) + "/"], check=False)
        print("downloaded to", dst); return
    if a.stage == "probe":
        i = budget(GPU, 1200 / 3600, "probe")
        ref = (ROOT / "results/parity_ref.npz").read_bytes()
        with modal.enable_output(), app.run():
            res = probe.with_options(gpu=GPU).remote(ref, TF32, int(os.environ.get("LABEL_BATCH", "16")))
        finished(i)
        print(json.dumps(res, indent=1))
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / f"results/modal_probe_{GPU}{'_tf32' if TF32 else ''}.json").write_text(json.dumps(res, indent=1))
        bad = {k: v for k, v in res["parity_db"].items() if v < 40}
        print("PARITY", "FAIL " + str(bad) if bad else "PASS (all > 40 dB vs CPU PyTorch)")
        return
    if a.stage == "prep":
        i = budget("cpu", 1.5, "prep")
        with modal.enable_output(), app.run(detach=True):
            print("meta.jsonl sha256 (sr16000):", prep.remote())
        finished(i)
        return
    hours = a.hours or (2.0 if a.stage == "label" else 24.0)
    i = budget(GPU, hours, a.stage)
    if a.stage == "label":
        with modal.enable_output(), app.run(detach=True):
            print(label.with_options(gpu=GPU, timeout=int(hours * 3600)).remote(int(os.environ.get("LABEL_BATCH", "16")), TF32))
        finished(i)
        return
    with modal.enable_output(), app.run(detach=True):
        call = train.with_options(gpu=GPU, timeout=int(hours * 3600)).spawn(DEPTH, STEPS, BATCH, a.extra.split(), TF32)
    state.write_text(json.dumps({"call": call.object_id, "t": time.time(), "gpu": GPU, "hours": hours}))
    print(f"training launched (detached) on {GPU}, call {call.object_id}. Check: bash run.sh modal status")


if __name__ == "__main__":
    main()
