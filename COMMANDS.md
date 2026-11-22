# RE-USE Lite: your commands

Everything below is copy-paste ready for **Terminal** on this Mac. Setup, the data and all the
conversion paths are already done and tested (see "What was tested" at the end). What is left is
the long GPU work, which you run yourself:

| Step | Command | Time on this M4 Air (measured) |
|---|---|---|
| 1. Label | `bash run.sh label` | ~7.5 h (22,944 crops at ~1.15 s each) |
| 2. Train | `bash run.sh train` | ~17.5 h (30,000 steps at ~2.1 s); up to ~22 h if the Air throttles |
| 3. Check | `bash run.sh all-after-train` | ~1.5 h (eval ~1.3 h, then export, convert, format checks) |
| 4. Publish | `bash run.sh publish` | ~5 min |

Before each long step:
- Plug in the charger and keep the lid open. `caffeinate` keeps the Mac awake on power, but closing the lid still sleeps it.
- Don't run another training job (heretic, MLX LoRA, other sessions) at the same time. `run.sh` refuses to start if it sees one; `FORCE=1` overrides.
- If anything stops it (crash, reboot, Ctrl+C), **run the same command again**. Labelling continues from the last saved crop and training from the last checkpoint (saved every 500 steps).

## 1. Label (the teacher enhances every training crop)

```sh
cd /Volumes/Untitled/re-use-lite
bash run.sh label
```

It prints progress and the hours left per sample rate. When it finishes it prints `labelled ... crops`.
It writes `data/train*/sr*/teacher.f32` (about 2.2 GB).

## 2. Train (distil 30 blocks into 4)

```sh
cd /Volumes/Untitled/re-use-lite
bash run.sh train
```

A log line every 50 steps shows the loss, seconds per step and `eta`. Every 1,000 steps it measures the
validation loss and saves the best model to `ckpt/run-d4/best/`. Check on it from another Terminal tab:

```sh
cd /Volumes/Untitled/re-use-lite && bash run.sh status
```

**Optional early check (recommended around step 5,000, ~10 h in):** press Ctrl+C in the training
window, then run:

```sh
cd /Volumes/Untitled/re-use-lite && bash run.sh peek
```

It scores the current best model on 100 test clips (~15 min) and prints the same table as the final gate, with a
ratio column. Ratios already near 0.9 or above are a good sign; much lower ones mean you can stop early and
ask me before spending the rest of the day. Then continue training with `bash run.sh train` (it resumes).
The peek never counts as the publishing gate.

Options (put them in front of the command, and use the same ones when resuming):
- `STEPS=20000 bash run.sh train` for a shorter schedule (~12 h).
- `DEPTH=3 bash run.sh train` trains a 3-block student instead (6.0x faster than the teacher at 16 kHz, but lower quality). It uses its own folder, `ckpt/run-d3`. Use the same `DEPTH=3` in front of every later command.
- `BATCH=4` is not recommended: it needs ~11 GB of GPU memory for the same speed.

## 3. Evaluate, export, convert, check

```sh
cd /Volumes/Untitled/re-use-lite
bash run.sh all-after-train
```

This runs four stages in a row (each can also be run alone):

1. `bash run.sh eval`: the quality gate on the 824 VoiceBank-DEMAND test clips (16 kHz, 8 kHz and an extra-degraded copy). It prints a table and `GATE: PASS` or `GATE: FAIL`, and writes `results/gate.json`. PASS means every PESQ, ESTOI and DNSMOS score is at least 95% of the teacher's on all three sets, and the student is at least 5x faster.
2. `bash run.sh export`: builds `out/normal` (PyTorch weights, inference.py, LICENSE, NOTICE, before/after samples).
3. `bash run.sh convert`: builds `out/mlx`, `out/gguf/re-use-lite-f32.gguf` and `-f16.gguf`, and the browser app `out/web`.
4. `bash run.sh check`: runs the same 9 clips through MLX, both GGUFs and the browser app (headless Chromium with WebGPU), compares each with PyTorch, and prints `CHECK PASS` or `CHECK FAIL`.

Try the browser app yourself: double-click `out/web/start.command` (it opens http://localhost:8737 in your browser).
It needs Brave, Chrome or Edge. It processes at 16 kHz by default (the rate the gate tests); the Rate menu offers 8, 24 and 48 kHz or the WAV's own rate. Note: your default browser for links is Dia, so if it opens there, paste the address into Brave.

Speed is part of the gate: at least 5x faster at 16 kHz on the whole test set. The bench gave 5.04x and
4 real clips gave 5.6x, so the margin is thin. The gate can fail on speed alone, for example if
something else is using the GPU during eval, so close other heavy apps while it runs. At 8 kHz the
speed-up is ~4.9x; the cards say so.

If the gate fails, nothing gets published. The options then are training longer (`STEPS=45000 bash run.sh train` continues the same run with a longer schedule) or a deeper student (`DEPTH=5`, 4.4x faster, so it can't pass the 5x speed rule as it stands). Ask me and I'll look at `results/gate.json` with you.

## 4. Publish

Optionally preview first. This writes every model card to `out/publish/*/README.md` and uploads nothing:

```sh
cd /Volumes/Untitled/re-use-lite
DRY_RUN=1 bash run.sh publish
```

Then publish for real:

```sh
cd /Volumes/Untitled/re-use-lite
bash run.sh publish
```

`publish` refuses unless both `results/gate.json` and `out/check/report.json` say PASS. It then:
- creates and uploads (public) `Lucasgitguy-co/RE-USE-Lite`, `-MLX`, `-GGUF` and the static Space `Lucasgitguy-co/RE-USE-Lite-web` (COOP/COEP headers are set in the Space README);
- deploys the same web app to Vercel (project `re-use-lite`);
- downloads all three model repos into an empty folder and checks them again;
- opens the live Space and Vercel pages in headless Chromium and runs the "Try the sample" button.

Uploads skip the `._*` files that macOS writes on this drive.

## What was tested before handing over (2026-11-22)

- Scan math: Metal and CPU scans against a float64 reference (`bash run.sh test`).
- Teacher port vs the official NVIDIA Space: 54.8 dB SI-SDR, re-checked with torch 2.15.
- Speed bench (`results/bench_mps.json`): at 16 kHz, 4 blocks run 5.04x faster than 30 blocks (8 kHz: 4.92x). On real test clips eval measured 5.6x.
- GGUF converter: the teacher converted with `scripts/convert_gguf.py` has tensors byte-identical to the official `audio-cpp/RE-USE-GGUF` file, and its output is identical (SI-SDR = infinite) under the patched audio.cpp, 62.6 dB vs PyTorch.
- Every format, using the untrained 4-block student, against PyTorch on CPU (9 clips at 8/16/44.1/48 kHz). Worst file: MLX 45.8 dB, GGUF f32 50.2 dB, GGUF f16 46.8 dB, browser (WebGPU) 47.6 dB. GGUF q8_0 failed (25 to 35 dB), so it is not shipped.
- Smoke training run: 12 clips labelled; 2 steps, stop, resume, 1 more step; checkpoint, best/ and last/ written. Batch 2 used 6.2 GB at 2.1 s per step.
- eval.py on 4 clips: the gate correctly fails the untrained student.
- Browser: a 50 s recording through the file picker (chunked path), and the "open in its own tab" fallback when embedded in an iframe.
- Browser UI: the "Try the sample" button in headless Chromium, both through the test server and through `serve.py` (the server `start.command` uses).
- Not tested: the real label/train runs, and publishing (it needs the trained model; `DRY_RUN=1` was run).

## Files

- `run.sh`: every stage. `PROGRESS.md`: state and numbers. `COMMANDS.md`: this file.
- `scripts/`: `prep_data.py`, `label.py`, `distill.py`, `eval.py`, `export.py`, `convert_gguf.py`, `check_formats.py`, `mlx_enhance.py`, `web_check.mjs`, `build_web.sh`, `build_browser.sh`, `publish.py`, `init_student.py`, plus the older tests and bench.
- `patches/audiocpp-reuse-depth.patch`: lets audio.cpp load 1 to 30 blocks (stock audio.cpp requires exactly 30).
- `web/`: the browser page. `release/`: inference.py, requirements and NOTICE for the model repos.
- Tools (venvs, audio.cpp, Emscripten, Playwright) live in `~/re-use-lite-tools`, because this exFAT drive can't hold them.
