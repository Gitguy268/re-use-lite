"""Publish RE-USE Lite: 3 model repos + a static Space on Hugging Face, and a Vercel mirror.

usage: publish.py [--dry-run] [--org Lucasgitguy-co] [--no-vercel]
Refuses unless results/gate.json and out/check/report.json both say PASS.
--dry-run builds the staging folders and cards in out/publish/ and uploads nothing (it also
works without a passed gate, so the cards can be read before training finishes).
After uploading it downloads every model repo into an empty folder and re-checks it.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reuse_lite.fsutil import rmtree
from reuse_lite.model import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--org", default="Lucasgitguy-co")
ap.add_argument("--name", default="RE-USE-Lite")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--no-vercel", action="store_true")
a = ap.parse_args()
TOOLS = os.environ.get("TOOLS", os.path.expanduser("~/re-use-lite-tools"))
J = lambda *p: os.path.join(ROOT, *p)
IGN = ["._*", ".DS_Store", "__pycache__", "*.pyc"]

gate = json.load(open(J("results/gate.json"))) if os.path.isfile(J("results/gate.json")) else None
check = json.load(open(J("out/check/report.json"))) if os.path.isfile(J("out/check/report.json")) else None
ok_gate = bool(gate and gate["gate"]["PASS"])
ok_check = bool(check and check["PASS"] and {"mlx", "gguf-re-use-lite-f32", "browser"} <= set(check["formats"]))
if not a.dry_run and not (ok_gate and ok_check):
    raise SystemExit(f"Not publishing: quality gate PASS={ok_gate}, format check PASS={ok_check}. "
                     "See results/gate.json and out/check/report.json.")
for p in ("out/normal/model.safetensors", "out/mlx/model.safetensors", "out/gguf/re-use-lite-f32.gguf", "out/web/index.html"):
    if not os.path.isfile(J(p)):
        raise SystemExit(f"missing {p}; run export/convert first")

cfg = json.load(open(J("out/normal/config.json")))
depth = cfg["model_cfg"]["num_tfmamba"]
R = {k: f"{a.org}/{a.name}{s}" for k, s in (("normal", ""), ("mlx", "-MLX"), ("gguf", "-GGUF"), ("space", "-web"))}
URL = {k: f"https://huggingface.co/{'spaces/' if k == 'space' else ''}{v}" for k, v in R.items()}
VERCEL_URL = None  # set to the real URL after a deploy that passes the live browser check
SPACE_URL = f"https://{a.org.lower()}-{(a.name + '-web').lower()}.static.hf.space/"  # direct page (not the iframe)


def live_check(url):
    r = subprocess.run(["node", J("scripts/web_check.mjs"), url],
                       env={**os.environ, "PW_DIR": os.path.join(TOOLS, "webtest/node_modules")})
    return r.returncode == 0


def deploy_vercel():
    """Deploy out/web from a folder named re-use-lite (so the project gets that name); return its URL."""
    import re
    vd = os.path.join(tempfile.mkdtemp(), "re-use-lite")
    shutil.copytree(J("out/web"), vd, ignore=shutil.ignore_patterns(*IGN, "start.command", "serve.py"))
    r = subprocess.run(["vercel", "deploy", "--prod", "--yes"], cwd=vd, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout[-1500:])
    if r.returncode:
        print("Vercel deploy failed; continuing with Hugging Face only. Retry later: cd out/web && vercel deploy --prod")
        return None
    urls = re.findall(r"https://[A-Za-z0-9.-]+\.vercel\.app", r.stdout)
    alias = [u for u in urls if u.startswith("https://re-use-lite")]
    return (alias or urls or [None])[-1]


if not a.dry_run and not a.no_vercel:
    v = deploy_vercel()
    if v and live_check(v + "/"):
        VERCEL_URL = v
    else:
        print("Vercel page did not pass the live check (failed deploy, or a login wall from Deployment Protection);"
              " the cards will not link it.")
gguf_types = [t for t in ("f32", "f16")
              if check is None or check["formats"].get(f"gguf-re-use-lite-{t}", {}).get("pass")]


def results_md():
    if not gate:
        return "_Evaluation has not run yet._\n"
    s = gate["sets"]
    lines = ["| Set | Metric | Noisy | Teacher (30 blocks) | **RE-USE Lite** | Ratio |", "|---|---|---:|---:|---:|---:|"]
    names = {"vbd16": "VoiceBank-DEMAND test, 16 kHz", "vbd8": "same, 8 kHz", "degraded": "same + reverb/low-pass/clipping"}
    for k, v in s.items():
        for m, lab in (("pesq", "PESQ" + (" WB" if v["sr"] == 16000 else " NB")), ("estoi", "ESTOI"), ("sisdr", "SI-SDR (dB)"),
                       ("ovrl", "DNSMOS OVRL"), ("sig", "DNSMOS SIG"), ("bak", "DNSMOS BAK")):
            r = v["student"][m] / v["teacher"][m] if v["teacher"][m] else float("nan")
            lines.append(f"| {names[k]} | {lab} | {v['noisy'][m]:.3f} | {v['teacher'][m]:.3f} | **{v['student'][m]:.3f}** | {r:.3f} |")
    lines.append("")
    sp = ", ".join(f"{k} {v['speedup']:.2f}x" for k, v in s.items())
    lines.append(f"Speed (PyTorch, Apple M4 GPU, whole test set, same code and device for both): **{gate['speedup_vbd16']:.2f}x faster "
                 f"at 16 kHz** ({sp}). The gate requires at least 5x at 16 kHz; at 8 kHz the fixed encoder/decoder cost "
                 "weighs more, so the speed-up there can be just under 5x. "
                 f"Parameters: {gate['params']['student']:,} vs "
                 f"{gate['params']['teacher']:,} ({gate['params']['teacher'] / gate['params']['student']:.1f}x smaller).")
    return "\n".join(lines) + "\n"


def check_md():
    if not check:
        return ""
    rows = ["| Format | Worst file | Mean |", "|---|---:|---:|"]
    for k, v in check["formats"].items():
        rows.append(f"| {k} | {v['min_db']:.1f} dB | {v['mean_db']:.1f} dB |")
    return ("Each format was run on the same 9 clips (8, 16, 44.1 and 48 kHz) and compared with the PyTorch "
            "model on CPU (SI-SDR; higher is closer):\n\n" + "\n".join(rows) + "\n")


FRONT = """---
license: other
license_name: openmdw-1.1
license_link: LICENSE
base_model: nvidia/NVIDIA-NemotronLabs-RE-USE
base_model_relation: {rel}
pipeline_tag: audio-to-audio
{lib}tags:
- speech-enhancement
- denoising
- dereverberation
- declipping
- bandwidth-extension
- mamba
- distillation
{tags}datasets:
- JacobLinCool/VoiceBank-DEMAND-16k
language:
- en
---
"""
ABOUT = f"""RE-USE Lite is an **unofficial, distilled student** of
[NVIDIA NemotronLabs RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE), a universal speech
enhancement model (noise, reverb, clipping, bandwidth loss; 8 to 48 kHz input; quality measured at 8 and 16 kHz,
other rates checked only for agreement between formats). The student keeps the
teacher's encoder and decoders and **{depth} of its 30 TFMamba blocks**, then was trained on a Mac (Apple M4)
to match the teacher's output. It is not affiliated with or endorsed by NVIDIA.

| Version | Repo |
|---|---|
| PyTorch (safetensors) | [{R['normal']}]({URL['normal']}) |
| MLX (Apple silicon) | [{R['mlx']}]({URL['mlx']}) |
| GGUF (audio.cpp) | [{R['gguf']}]({URL['gguf']}) |
| Browser (WebGPU) | [{SPACE_URL}]({SPACE_URL}){(" · [" + VERCEL_URL + "](" + VERCEL_URL + ")") if VERCEL_URL else ""} ([Space page]({URL['space']})) |
"""
LIMITS = """## Limits

- Distillation data is VoiceBank-DEMAND (English read speech) with added reverb, low-pass and clipping, so
  other languages, singing and very noisy field recordings are less tested than in the teacher.
- Offline only (bidirectional): not for live streaming. Overlapping speakers may be suppressed.
- Speed numbers come from the PyTorch port with a pure-PyTorch Mamba (no CUDA `mamba_ssm`).

## How it was made

1. Student = teacher encoder + both decoders + TFMamba blocks {picked} (evenly spaced).
2. Teacher outputs on 2 random 24,000-sample crops per VoiceBank-DEMAND training clip, at 8/16/24/48 kHz, with
   synthetic reverb, low-pass and clipping added (code: [Gitguy268/re-use-lite](https://github.com/Gitguy268/re-use-lite)).
3. Loss: SEMamba-style magnitude, anti-wrapped phase, complex, time and consistency terms against the teacher,
   plus 0.3x the same against clean speech where valid. AdamW, cosine schedule, {steps} steps, batch {batch}.
4. Published only after the gate passed: every PESQ/ESTOI/DNSMOS score at least 95% of the teacher's on all
   three sets, and at least 5x faster at 16 kHz (PyTorch, Apple M4 GPU).

## License

OpenMDW-1.1, as the teacher. See `LICENSE` (with NVIDIA's copyright notice) and `NOTICE` (Apache-2.0 Mamba
port and audio.cpp, CC BY 4.0 VoiceBank-DEMAND).

## Citation

Please cite the RE-USE papers (from NVIDIA's model card):

```bibtex
@article{{fu2026rethinking,
  title={{Rethinking Training Targets, Architectures and Data Quality for Universal Speech Enhancement}},
  author={{Fu, Szu-Wei and Chao, Rong and Yang, Xuesong and Huang, Sung-Feng and Zezario, Ryandhimas E and Nasretdinov, Rauf and Juki{{\\'c}}, Ante and Tsao, Yu and Wang, Yu-Chiang Frank}},
  journal={{arXiv preprint arXiv:2603.02641}},
  year={{2026}}
}}

@article{{fu2026one,
  title={{One Model, Many Latencies: Universal Speech Enhancement for Diverse Real-Time Applications}},
  author={{Fu, Szu-Wei and Chao, Rong and Yang, Xuesong and Huang, Sung-Feng and Juki{{\\'c}}, Ante and Tsao, Yu and Wang, Yu-Chiang Frank}},
  journal={{arXiv preprint arXiv:2606.25621}},
  year={{2026}}
}}
```
"""
run_args = {}
for cand in (J(f"ckpt/run-d{depth}"),):
    try:
        import torch
        lt = open(os.path.join(cand, "latest.txt")).read().strip()
        ck = torch.load(os.path.join(cand, lt), map_location="cpu", weights_only=False)
        run_args = dict(picked=ck["picked"], steps=ck["step"], batch=ck["args"]["batch"])
    except Exception:
        pass
limits = LIMITS.format(picked=run_args.get("picked", "?"), steps=run_args.get("steps", "?"), batch=run_args.get("batch", "?"))

cards = {
    "normal": FRONT.format(rel="finetune", lib="", tags="- pytorch\n") + f"""
# RE-USE Lite ({depth} blocks)

{ABOUT}
## Use (PyTorch: CUDA, Apple GPU or CPU)

```bash
hf download {R['normal']} --local-dir RE-USE-Lite
cd RE-USE-Lite && pip install -r requirements.txt
python inference.py noisy.wav -o enhanced/          # files or folders; keeps rate, channels, length
```

`samples/` has before/after examples (`*_enhanced.wav`).

## Results

{results_md()}
{check_md()}
{limits}""",
    "mlx": FRONT.format(rel="quantized", lib="library_name: mlx-audio\n", tags="- mlx\n- apple-silicon\n") + f"""
# RE-USE Lite for MLX ({depth} blocks, float32)

{ABOUT}
## Use

```bash
pip install "mlx-audio==0.6.6"
python -m mlx_audio.sts.generate --model {R['mlx']} --audio noisy.wav --output-path enhanced.wav
```

```python
from mlx_audio.sts import load
from mlx_audio import audio_io
model = load("{R['mlx']}", strict=True)
result = model.generate("noisy.wav")            # 5 s chunks by default; chunk_seconds=None for one pass
audio_io.write("enhanced.wav", result.audio, result.sample_rate)
```

Uses mlx-audio's `nemotron_reuse` model, which reads the block count from `config.json`. Kept in float32
(the model is small and the scan is precision sensitive).

## Results

{results_md()}
{check_md()}
{limits}""",
    "gguf": FRONT.format(rel="quantized", lib="library_name: audio.cpp\n", tags="- gguf\n- audio.cpp\n") + f"""
# RE-USE Lite GGUF ({depth} blocks)

{ABOUT}
## Files

| File | Type | Notes |
|---|---|---|
""" + "".join(f"| `re-use-lite-{t}.gguf` | {t.upper()} | {'recommended' if t == 'f32' else 'half size; passed the same check (> 30 dB vs PyTorch)'} |\n" for t in gguf_types) + f"""
Q8_0 was tried and left out: it drifted too far from the PyTorch output (about 25 to 35 dB).

## Use (audio.cpp)

Stock audio.cpp only loads RE-USE with exactly 30 blocks, so this needs audio.cpp at commit
`a6e32214e762aca6126eb7206a5497f11d064303` plus the 5-line `audiocpp-reuse-depth.patch` in this repo:

```bash
git clone https://github.com/0xShug0/audio.cpp && cd audio.cpp
git checkout a6e32214e762aca6126eb7206a5497f11d064303 && git submodule update --init --recursive
git apply /path/to/audiocpp-reuse-depth.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=reuse
cmake --build build --target audiocpp_cli -j 8
build/bin/audiocpp_cli --task s2s --family reuse --model re-use-lite-f32.gguf --backend cpu \\
  --audio noisy.wav --out enhanced.wav
```

The GGUF embeds its config. Input 8 to 48 kHz; output keeps rate, channels and length.

## Results

{results_md()}
{check_md()}
{limits}""",
    "space": f"""---
title: RE-USE Lite
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: other
short_description: Speech enhancement in your browser (WebGPU, local)
models:
- {R['normal']}
- {R['gguf']}
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# RE-USE Lite in the browser

Cleans up speech recordings locally on your GPU with [RE-USE Lite]({URL['normal']}), an unofficial
distilled student of NVIDIA NemotronLabs RE-USE, run by [audio.cpp](https://github.com/0xShug0/audio.cpp)
compiled to WebAssembly + WebGPU. Needs Chrome, Edge or Brave (WebGPU with shader-f16 and JSPI).
Licences are in `licenses/`. Not affiliated with NVIDIA.
""",
}

# ---------------- stage ----------------
stage = J("out/publish"); rmtree(stage)
dirs = {k: os.path.join(stage, k) for k in R}
shutil.copytree(J("out/normal"), dirs["normal"], ignore=shutil.ignore_patterns(*IGN))
shutil.copytree(J("out/mlx"), dirs["mlx"], ignore=shutil.ignore_patterns(*IGN, "README.md"))
os.makedirs(dirs["gguf"])
for t in gguf_types:
    shutil.copyfile(J(f"out/gguf/re-use-lite-{t}.gguf"), os.path.join(dirs["gguf"], f"re-use-lite-{t}.gguf"))
for f, src in (("LICENSE", J("vendor/LICENSE")), ("NOTICE", J("release/NOTICE")),
               ("audiocpp-reuse-depth.patch", J("patches/audiocpp-reuse-depth.patch"))):
    shutil.copyfile(src, os.path.join(dirs["gguf"], f))
for k in ("normal", "mlx"):
    for f, src in (("LICENSE", J("vendor/LICENSE")), ("NOTICE", J("release/NOTICE"))):
        shutil.copyfile(src, os.path.join(dirs[k], f))
shutil.copytree(J("out/web"), dirs["space"], ignore=shutil.ignore_patterns(*IGN, "start.command", "serve.py", "vercel.json"))
for k in ("normal", "mlx", "gguf"):
    for f in ("results/gate.json", "out/check/report.json"):
        if os.path.isfile(J(f)):
            os.makedirs(os.path.join(dirs[k], "eval"), exist_ok=True)
            shutil.copyfile(J(f), os.path.join(dirs[k], "eval", os.path.basename(f)))
for k, text in cards.items():
    open(os.path.join(dirs[k], "README.md"), "w").write(text)
print("staged:", {k: sorted(os.listdir(v)) for k, v in dirs.items()})
if a.dry_run:
    print(f"DRY RUN: nothing uploaded. Gate PASS={ok_gate}, check PASS={ok_check}. Cards are in {stage}/*/README.md")
    sys.exit(0)

# ---------------- upload ----------------
from huggingface_hub import HfApi, snapshot_download  # noqa: E402
api = HfApi()
for k in ("normal", "mlx", "gguf"):
    api.create_repo(R[k], repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=R[k], repo_type="model", folder_path=dirs[k], ignore_patterns=IGN,
                      commit_message="RE-USE Lite release")
    print("uploaded", URL[k])
api.create_repo(R["space"], repo_type="space", space_sdk="static", private=False, exist_ok=True)
api.upload_folder(repo_id=R["space"], repo_type="space", folder_path=dirs["space"], ignore_patterns=IGN,
                  commit_message="RE-USE Lite web app")
print("uploaded", URL["space"])

# ---------------- verify from the Hub ----------------
work = tempfile.mkdtemp(prefix="re-use-lite-verify-")
dl = {k: snapshot_download(R[k], local_dir=os.path.join(work, k)) for k in ("normal", "mlx", "gguf")}
cmd = [sys.executable, J("scripts/check_formats.py"), "--normal", dl["normal"], "--mlx", dl["mlx"],
       "--gguf", os.path.join(dl["gguf"], "re-use-lite-f32.gguf"), "--work", os.path.join(work, "check")]
r = subprocess.run(cmd)
print("Hub download check:", "PASS" if r.returncode == 0 else "FAIL", f"(work dir {work})")
# live browser check: the Space serves from <owner>-<name>.static.hf.space
ok = live_check(SPACE_URL)
print("live browser check", SPACE_URL, "PASS" if ok else "FAIL (a Space can take a minute to build; rerun: "
      f"PW_DIR=$TOOLS/webtest/node_modules node scripts/web_check.mjs {SPACE_URL})")
print("Done:", *URL.values(), SPACE_URL, VERCEL_URL or "(no Vercel link)", sep="\n  ")
