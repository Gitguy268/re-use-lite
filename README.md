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
| Pure-PyTorch Mamba | `reuse_lite/mamba_torch.py` | parameter-compatible with `mamba_ssm.Mamba`; chunked selective scan in log space; CPU, CUDA and MPS |
| Metal scan | `reuse_lite/scan_metal.py` | fused selective scan for Apple GPUs (forward and backward) through `torch.mps.compile_shader` |
| Loader | `reuse_lite/model.py` | loads the teacher from `vendor/`, or builds a student from evenly spaced teacher blocks |
| Checks | `scripts/test_scan.py`, `scripts/test_metal.py` | scan against a float64 sequential reference |
| Parity | `scripts/parity.py` | local port on CPU and MPS against the official Hugging Face Space output |
| Speed | `scripts/bench.py`, `scripts/bench_scan.py` | real-time factor of teacher and students at several depths and sample rates |

`model.py` uses the real `mamba_ssm` when it imports and falls back to the PyTorch version otherwise. Set `REUSE_TORCH_MAMBA=1` to force the fallback.

## Quick start

```sh
git clone https://github.com/Gitguy268/re-use-lite.git
cd re-use-lite
uv venv --python 3.12 .venv
uv pip install --python .venv torch safetensors soundfile numpy
hf download nvidia/NVIDIA-NemotronLabs-RE-USE --local-dir vendor   # teacher weights and its model code
.venv/bin/python scripts/test_scan.py
.venv/bin/python scripts/parity.py
.venv/bin/python scripts/bench.py --device mps --depths 30,8,6,5,4
```

## Gates

- [x] **0a, scan parity:** relative error of about 4e-5 or less against a float64 sequential scan, on CPU and MPS, at lengths 161, 200, 323 and 400.
- [x] **0b, teacher parity:** [`results/parity/parity.json`](results/parity/parity.json) shows 54.8 dB SI-SDR between this port and the official Space on `mic_test2.wav` (target above 30 dB), and 67.8 dB between CPU and MPS.
- [ ] **1, speed profile:** pick the student depth.
- [ ] **2, distillation** and the quality comparison.

## Status

The scan, Metal kernel and teacher parity work. The speed profile and distillation have not run, so there is no student model yet. Nothing has been published to Hugging Face.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

- The repository is under the [MIT License](LICENSE).
- `reuse_lite/mamba_torch.py` and `reuse_lite/scan_metal.py` are under the [Apache License 2.0](LICENSES/Apache-2.0.txt), because their math follows `mamba_ssm` (Copyright (c) 2023 Tri Dao, Albert Gu, Apache-2.0). Each file says so in its SPDX header.
- The teacher's weights and code in `vendor/` are not in this repository. They are NVIDIA's, under the OpenMDW-1.1 licence.

See [NOTICE](NOTICE). This project is not affiliated with NVIDIA.
