<div align="center">

# RE-USE Lite

Work toward a smaller, faster student of NVIDIA's RE-USE speech enhancement model, with a pure-PyTorch Mamba that runs on Apple GPUs.

[Goal](#goal) · [What works](#what-works) · [Quick start](#quick-start) · [Gates](#gates) · [Status](#status) · [License](#license)

![Status](https://img.shields.io/badge/status-early%20research-e17055) ![Python](https://img.shields.io/badge/python-3.12-3776ab) ![Devices](https://img.shields.io/badge/devices-CPU%20%7C%20MPS%20%7C%20CUDA-636e72) ![License](https://img.shields.io/badge/license-MIT%20%2B%20Apache--2.0-2d3436)

</div>

## Goal

[nvidia/NVIDIA-NemotronLabs-RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE) is a 9.6M-parameter universal speech enhancement model built from 30 TF-Mamba blocks. Its official code needs `mamba_ssm`, which only has CUDA kernels.

The goal is a distilled student that is **at least 5× faster** with PESQ, ESTOI and DNSMOS **within 5%** of the teacher. A model will only be published if it meets that target.

## What works

| Part | File | What it does |
|---|---|---|
| Pure-PyTorch Mamba | `reuse_lite/mamba_torch.py` | parameter-compatible with `mamba_ssm.Mamba`; CPU, CUDA and MPS |
| Metal scan | `reuse_lite/scan_metal.py` | fused selective scan for Apple GPUs (forward and backward) through `torch.mps.compile_shader` |
| Loader | `reuse_lite/model.py` | teacher, student from evenly spaced teacher blocks, batched enhance |
| Data | `scripts/prep_data.py` | VoiceBank-DEMAND crops at 8/16/24/48 kHz with reverb, low-pass and clipping |
| Distillation | `scripts/label.py`, `scripts/distill.py` | teacher labels, then a resumable MPS training loop |
| Gate | `scripts/eval.py` | PESQ, ESTOI, SI-SDR, DNSMOS and speed, student vs teacher |
| Formats | `scripts/export.py`, `scripts/convert_gguf.py`, `scripts/build_web.sh` | PyTorch release folder, MLX (mlx-audio), GGUF (audio.cpp) and a WebGPU browser app |
| Checks | `scripts/check_formats.py`, `scripts/web_check.mjs` | every format against PyTorch, including the browser in headless Chromium |
| audio.cpp patch | `patches/audiocpp-reuse-depth.patch` | lets audio.cpp's `reuse` family load 1 to 30 blocks |

## Quick start

On an Apple-silicon Mac with [uv](https://github.com/astral-sh/uv), Node and CMake:

```sh
git clone https://github.com/Gitguy268/re-use-lite.git && cd re-use-lite
bash run.sh setup      # venvs, audio.cpp + patch, Emscripten, Playwright, teacher, VoiceBank-DEMAND
bash run.sh test && bash run.sh prep
bash run.sh label      # ~7.5 h on an M4 Air
bash run.sh train      # ~17.5 h, resumable
bash run.sh all-after-train && bash run.sh publish
```

[COMMANDS.md](COMMANDS.md) explains each step. `TOOLS` (default `~/re-use-lite-tools`) holds venvs and builds, so the repo itself can live on an exFAT drive.

## Gates

- [x] **0a, scan parity:** relative error of about 4e-5 or less against a float64 sequential scan, on CPU and MPS.
- [x] **0b, teacher parity:** [`results/parity/parity.json`](results/parity/parity.json) shows 54.8 dB SI-SDR between this port and the official Space on `mic_test2.wav`.
- [x] **1, speed profile:** [`results/bench_mps.json`](results/bench_mps.json). At 16 kHz on an M4 GPU, 4 blocks run 5.04x faster than 30 (6 blocks 3.9x, 3 blocks 6.0x). Depth 4 was chosen.
- [x] **Formats:** with the untrained 4-block student, MLX, GGUF f32/f16 and the browser build all stay above 45 dB SI-SDR against PyTorch. GGUF Q8_0 does not, so it is not shipped.
- [ ] **2, distillation** and the quality comparison.

## Status

Everything up to training is built and tested. Labelling and training have not run yet, so there is no trained student and nothing is on Hugging Face.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

- The repository is under the [MIT License](LICENSE).
- `scripts/convert_gguf.py` and `web/worker.js` adapt audio.cpp code (Copyright 2026 ShugoAI LLC) and are Apache-2.0, as their SPDX headers say.
- `reuse_lite/mamba_torch.py` and `reuse_lite/scan_metal.py` are under the [Apache License 2.0](LICENSES/Apache-2.0.txt), because their math follows `mamba_ssm` (Copyright (c) 2023 Tri Dao, Albert Gu, Apache-2.0). Each file says so in its SPDX header.
- The teacher's weights and code in `vendor/` are not in this repository. They are NVIDIA's, under the OpenMDW-1.1 licence.

See [NOTICE](NOTICE). This project is not affiliated with NVIDIA.
