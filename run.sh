#!/usr/bin/env bash
# RE-USE Lite: distil NVIDIA NemotronLabs RE-USE into a faster student, convert, check, publish.
# Usage: bash run.sh <setup|test|bench|prep|label|train|status|peek|eval|export|convert|check|publish|all-after-train|modal ...>
# See COMMANDS.md for the full walkthrough. Long stages are resumable: run the same command again.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"            # /Volumes/Untitled/re-use-lite (data, checkpoints, outputs)
TOOLS="${TOOLS:-$HOME/re-use-lite-tools}"         # venvs, audio.cpp, emsdk, playwright (exFAT can't hold them)
export TOOLS
PY="$TOOLS/.venv-torch/bin/python"
MLXPY="$TOOLS/.venv-mlx/bin/python"
GGUFPY="$TOOLS/.venv-gguf/bin/python"
HF="$TOOLS/.venv-torch/bin/hf"
ACPP="$TOOLS/ext/audio.cpp"
ACPP_REV=a6e32214e762aca6126eb7206a5497f11d064303
EMSDK_VER=6.0.13
PW_VER=1.64.1
TEACHER_REV=31f82f2efad1095034492f236e8dac9239ba713b
DEPTH="${DEPTH:-4}"
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-2}"
RUN="$ROOT/ckpt/run-d$DEPTH"
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$ROOT"
mkdir -p logs

busy_check() {
  # Two GPU jobs on 16 GB swap each other to a crawl; refuse unless FORCE=1.
  local others
  others="$(pgrep -fl "distill.py|label.py|mlx_lm|lora|train.py|heretic" | grep -v "$$" || true)"
  if [ -n "$others" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "Another ML job is running:" >&2; echo "$others" >&2
    echo "Wait for it to finish, or run with FORCE=1." >&2
    exit 1
  fi
}

need() { [ -e "$1" ] || { echo "Missing $1 -- $2" >&2; exit 1; }; }

case "${1:-}" in
setup)
  command -v uv >/dev/null || { echo "Install uv first: brew install uv" >&2; exit 1; }
  mkdir -p "$TOOLS/ext"
  cd "$TOOLS"
  [ -x .venv-torch/bin/python ] || { uv venv -q --python 3.12 .venv-torch; VIRTUAL_ENV=.venv-torch uv pip install -r "$ROOT/requirements/torch.lock.txt"; }
  [ -x .venv-mlx/bin/python ] || { uv venv -q --python 3.11 .venv-mlx; VIRTUAL_ENV=.venv-mlx uv pip install -r "$ROOT/requirements/mlx.lock.txt"; }
  [ -x .venv-gguf/bin/python ] || { uv venv -q --python 3.12 .venv-gguf; VIRTUAL_ENV=.venv-gguf uv pip install -r "$ROOT/requirements/gguf.lock.txt"; }
  cd "$TOOLS/ext"
  [ -d audio.cpp ] || git clone https://github.com/0xShug0/audio.cpp.git
  git -C audio.cpp checkout -q "$ACPP_REV" && git -C audio.cpp submodule update --init --recursive -q
  if git -C audio.cpp apply --check "$ROOT/patches/audiocpp-reuse-depth.patch" 2>/dev/null; then
    git -C audio.cpp apply "$ROOT/patches/audiocpp-reuse-depth.patch"
  fi
  grep -q "blocks - 1" audio.cpp/src/community_models/reuse/runtime.cpp || { echo "audio.cpp patch not applied" >&2; exit 1; }
  cmake -S audio.cpp -B audio.cpp/build/cpu -DCMAKE_BUILD_TYPE=Release -DENGINE_ENABLE_CUDA=OFF \
    -DENGINE_ENABLE_VULKAN=OFF -DENGINE_ENABLE_METAL=OFF -DENGINE_ENABLE_OPENMP=OFF -DGGML_OPENMP=OFF \
    -DAUDIOCPP_DEPLOYMENT_BUILD=ON -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=reuse -DCMAKE_OSX_ARCHITECTURES=arm64
  cmake --build audio.cpp/build/cpu --parallel 6 --target audiocpp_cli audiocpp_gguf
  [ -d emsdk ] || git clone https://github.com/emscripten-core/emsdk.git
  (cd emsdk && ./emsdk install "$EMSDK_VER" && ./emsdk activate "$EMSDK_VER")
  [ -f audio.cpp/build/browser/bin/audiocpp_cli.wasm ] || bash "$ROOT/scripts/build_browser.sh"
  mkdir -p "$TOOLS/webtest" && cd "$TOOLS/webtest"
  [ -f package.json ] || npm init -y >/dev/null
  [ -d node_modules/playwright ] || { npm i -s "playwright@$PW_VER"; npx playwright install chromium; }
  cd "$ROOT"
  [ -f vendor/model.safetensors ] || "$HF" download nvidia/NVIDIA-NemotronLabs-RE-USE --revision "$TEACHER_REV" --local-dir vendor
  [ -f data/vbd/data/test-00000-of-00001.parquet ] && ls data/vbd/data/train-*.parquet >/dev/null 2>&1 || \
    "$HF" download JacobLinCool/VoiceBank-DEMAND-16k --repo-type dataset --include "data/*" --local-dir data/vbd
  echo "setup done"
  ;;
test)
  "$PY" scripts/test_scan.py
  "$PY" scripts/test_metal.py
  ;;
bench)
  busy_check
  caffeinate -i "$PY" scripts/bench.py --device mps --depths 30,8,6,5,4,3 --srs 16000,8000 --reps 3
  ;;
prep)
  [ -f data/train/DONE ] && [ -f data/train_val/DONE ] && { echo "data/train already built"; exit 0; }
  "$PY" scripts/prep_data.py --parquet 'data/vbd/data/train-*.parquet' --out data/train --variants 2 --val 100
  ;;
label)
  busy_check
  need data/train/DONE "run: bash run.sh prep"
  caffeinate -dimsu "$PY" scripts/label.py --data data/train_val --batch 4 2>&1 | tee -a logs/label.log
  caffeinate -dimsu "$PY" scripts/label.py --data data/train --batch 4 2>&1 | tee -a logs/label.log
  ;;
train)
  busy_check
  need data/train/sr16000/teacher.f32 "run: bash run.sh label"
  caffeinate -dimsu "$PY" scripts/distill.py --data data/train --val data/train_val --out "$RUN" \
    --depth "$DEPTH" --steps "$STEPS" --batch "$BATCH" --resume 2>&1 | tee -a logs/train.log
  ;;
status)
  tail -n 3 logs/label.log 2>/dev/null || true
  tail -n 5 "$RUN/train.log" 2>/dev/null || echo "no training log yet ($RUN/train.log)"
  pgrep -fl "label.py|distill.py" || echo "(no label/train process running)"
  ;;
peek)
  # Early go/no-go: stop training with Ctrl+C, run this (~15 min, 100 clips), then resume training.
  # Writes results/peek.json; it can never count as the publishing gate (not the full test set).
  busy_check
  need "$RUN/best/model.safetensors" "no best/ yet: train until the first validation (step 1000)"
  caffeinate -i "$PY" scripts/eval.py --student "$RUN/best" --limit 100 --out results/peek.json 2>&1 | tee logs/peek.log
  ;;
eval)
  busy_check
  need "$RUN/best/model.safetensors" "train first: bash run.sh train"
  caffeinate -i "$PY" scripts/eval.py --student "$RUN/best" --out results/gate.json 2>&1 | tee logs/eval.log
  ;;
export)
  need "$RUN/best/model.safetensors" "train first: bash run.sh train"
  "$PY" scripts/export.py --student "$RUN/best" --out out/normal
  ;;
convert)
  need out/normal/model.safetensors "run: bash run.sh export"
  rm -rf out/mlx out/gguf 2>/dev/null || true
  "$MLXPY" -m mlx_audio.convert --hf-path out/normal --mlx-path out/mlx --dtype float32
  cp release/NOTICE out/mlx/NOTICE
  for t in f32 f16; do
    "$GGUFPY" scripts/convert_gguf.py --source out/normal --output "out/gguf/re-use-lite-$t.gguf" --audiocpp "$ACPP" --type "$t"
  done
  bash scripts/build_web.sh out/gguf/re-use-lite-f32.gguf out/web
  ;;
check)
  need out/web/index.html "run: bash run.sh convert"
  "$PY" scripts/check_formats.py --normal out/normal --mlx out/mlx \
    --gguf out/gguf/re-use-lite-f32.gguf out/gguf/re-use-lite-f16.gguf --work out/check || true
  PW_DIR="$TOOLS/webtest/node_modules" node scripts/web_check.mjs out/web out/check/inputs out/check/browser
  "$PY" scripts/check_formats.py --normal out/normal --mlx out/mlx \
    --gguf out/gguf/re-use-lite-f32.gguf out/gguf/re-use-lite-f16.gguf --browser out/check/browser \
    --work out/check --keep-inputs
  ;;
all-after-train)
  bash "$0" eval && bash "$0" export && bash "$0" convert && bash "$0" check
  ;;
modal)
  # GPU stages on Modal (see modal_app.py): probe, prep, label, train, status, download
  shift
  DEPTH="$DEPTH" STEPS="$STEPS" BATCH="$BATCH" uv run --no-project --python 3.12 --with 'modal==1.6.4' \
    python modal_app.py "$@"
  ;;
publish)
  "$PY" scripts/publish.py ${DRY_RUN:+--dry-run}
  ;;
*)
  sed -n 2,4p "$0"; exit 1 ;;
esac
