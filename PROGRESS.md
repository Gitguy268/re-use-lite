# RE-USE Lite: progress

Plan: ~/.claude/plans/create-a-finetune-fuzzy-canyon.md
Goal: a student distilled from nvidia/NVIDIA-NemotronLabs-RE-USE (30 TFMamba blocks) that is ≥5× faster, with PESQ/ESTOI/DNSMOS within 5% of the teacher. Publish to Lucasgitguy-co/RE-USE-Lite only if that target is met.

## Facts
- Params: TSMamba 8.34M (277,888 per block), encoder 0.38M, mag decoder 0.44M, phase decoder 0.44M. Total 9.61M.
- The encoder downsamples time ×4 and freq ×2 before the Mamba blocks.
- STFT is sampling-frequency independent: n_fft/hop/win scale with sr (base 8 kHz: 320/40/320).
- Env: `.venv` (uv, Python 3.12). `reuse_lite/model.py` slots in `reuse_lite/mamba_torch.py` when `mamba_ssm` won't import (always on the Mac).

## Gates
- [x] 0a scan parity (`scripts/test_scan.py`): rel err ≤ 4e-5 vs float64 sequential, CPU + MPS, L = 161/200/323/400.
- [ ] 0b teacher on mic_test2.wav (44.1 kHz, 5.1 s), port vs official Space (`scripts/parity.py --space`), target SI-SDR > 30 dB
- [ ] 1 speed profile (`scripts/bench.py`), pick depth

## Log
- 2026-11-09: project set up, vendor downloaded, venv built, scan test passes. Disk was down to 10 GB free (other sessions writing), so keep VoiceBank as parquet and decode on the fly.
