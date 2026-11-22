# RE-USE Lite: progress

Plans: ~/.claude/plans/create-a-finetune-fuzzy-canyon.md (2026-11-09) and
~/.claude/plans/post-train-https-huggingface-co-nvidia-n-witty-blossom.md (2026-11-19, current).
**User-run commands: COMMANDS.md.**

Goal: a student distilled from nvidia/NVIDIA-NemotronLabs-RE-USE (30 TFMamba blocks, rev 31f82f2) that is
≥5x faster with PESQ/ESTOI/DNSMOS ≥95% of the teacher. Ship normal (PyTorch), MLX, GGUF (audio.cpp) and a
browser version (audio.cpp WASM + WebGPU) on HF Lucasgitguy-co, only if the gate passes. The user runs
label/train/publish themselves.

## Facts
- Params: TSMamba 277,888 per block; encoder + decoders 1.27M. 4 blocks = 2.38M (4.0x smaller).
- STFT scales with the sample rate (base 8 kHz: 320/40/320), so frames/s is 200 at every rate and the cost grows with the rate.
- Layout: data/checkpoints/outputs on /Volumes/Untitled/re-use-lite (exFAT); venvs, audio.cpp, emsdk and Playwright in ~/re-use-lite-tools.
- exFAT gotchas:
  - macOS writes `._*` files: skip them when listing and when uploading.
  - `shutil.rmtree` trips when the `._` twins vanish mid-walk, so use `reuse_lite/fsutil.rmtree`.
- launchd jobs can't read or write /Volumes/Untitled (TCC, removable volume; EX_CONFIG / "Operation not permitted"). So long jobs run in the user's Terminal under caffeinate, and they're resumable.
- PyTorch MPS vs CPU differ by ~45 dB SI-SDR (Metal scan float drift). Format checks therefore use **CPU** PyTorch as the reference.
- audio.cpp `reuse` hardcodes 30 blocks → `patches/audiocpp-reuse-depth.patch`.
  - Its own converter sha-locks the original nvidia/RE-USE checkpoint (87a4…), so we use `scripts/convert_gguf.py` instead.
  - audiocpp_gguf embeds every file in `--root`, so we stage only config.json.
- mlx-audio 0.6.6 `nemotron_reuse` reads num_tfmamba from config; `model_type: nemotron_reuse` is in our config.
- No Metal compiler (CLT only), so the native audio.cpp build is CPU-only. That's fine for conversion and checks.

## Gates
- [x] 0a scan parity (CPU + MPS).
- [x] 0b teacher port vs official Space: 54.8 dB (re-checked 2026-11-22 with torch 2.15).
- [x] 1 speed: `results/bench_mps.json`.
  - At 16 kHz: depth 6 3.94x, 5 4.38x, **4 5.04x**, 3 6.00x. At 8 kHz: depth 4 4.92x.
  - 48 kHz rows hit swap and are unreliable.
  - Chosen depth 4. eval on real clips measured 5.6x (4 clips).
- [x] conversions on the untrained 4-block student vs PyTorch CPU, worst of 9 files:
  - MLX 45.8 dB, GGUF f32 50.2 dB, GGUF f16 46.8 dB, browser 47.6 dB.
  - q8_0 25–35 dB, so it's dropped.
- [x] teacher GGUF from our converter: tensors byte-identical to audio-cpp/RE-USE-GGUF; output identical (inf dB) under the patched runtime.
- [x] browser extras:
  - 50.7 s recording through the real file picker (chunked path): full length, finite, 22 s.
  - In a non-isolated iframe, the page shows an "open in its own tab" link.
  - Default rate is 16 kHz, the gated rate.
- [x] smoke: prep → label → 2 steps → resume → 1 step. Batch 2 = 2.1 s/step, 6.2 GB; batch 4 = 4.2 s/step, 11 GB.
- [x] data built: data/train (11,472 clips × 2 crops) + data/train_val (100 clips × 2), 24,000 samples per crop.
- [x] Modal route (2026-11-22, user asked to run on Modal): `modal_app.py`, `bash run.sh modal <stage>`.
  - Image: torch 2.10 cu128 + pinned sha256 wheels mamba_ssm 2.3.2.post1, causal_conv1d 1.7.0. `REUSE_REQUIRE_MAMBA_SSM=1` makes a silent fallback to the torch scan impossible.
  - L4 probe: teacher parity vs CPU PyTorch 55.8/70.2/70.4/84.2 dB (fp32) and 55.7–74 dB (TF32). Label 0.22 s/crop at batch 16. Train batch 2: 0.41 s/step (ckpt), 0.35 (no ckpt), 0.29 (no ckpt + TF32).
  - prep on Modal CPU: identical rows, identical train meta.jsonl sha256 (c199921b…).
  - Budget rule: spend ≤ credit left − $1; worst case per launch in ~/.local/state/re-use-lite-modal/ledger.json. Credit at start $6.60.
- [ ] 2 label: Modal L4, fp32, batch 16 (`LABEL_BATCH=16 bash run.sh modal label --hours 1.8`)
- [ ] 3 train: Modal L4, batch 2, 30k steps, TF32, no grad ckpt (`REUSE_TF32=1 bash run.sh modal train --hours 2.9 --extra=--no-grad-ckpt`)
- [ ] 4 `bash run.sh modal download`, then eval gate, export, convert, check on the Mac: `bash run.sh all-after-train`
- [ ] 5 publish: `bash run.sh publish` (refuses unless gate + check PASS)

## Log
- 2026-11-09: project set up, scan test passes.
- 2026-11-19/22: moved to Untitled; tools rebuilt. Wrote data/label/distill/eval/export/convert/check/publish, the audio.cpp patch and the web app; everything up to training is verified (see Gates). Handed over to the user with COMMANDS.md.
