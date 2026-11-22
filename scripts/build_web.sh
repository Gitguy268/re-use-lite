#!/bin/bash
# Assemble the static web app: page + audio.cpp browser build + model GGUF + sample.
# usage: build_web.sh <model.gguf> <out_dir>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="${TOOLS:-$HOME/re-use-lite-tools}"
GGUF="$1"; OUT="$2"
BIN="$TOOLS/ext/audio.cpp/build/browser/bin"
[ -f "$BIN/audiocpp_cli.wasm" ] || { echo "browser build missing: run scripts/build_browser.sh" >&2; exit 1; }
[ -f "$GGUF" ] || { echo "no GGUF at $GGUF" >&2; exit 1; }
rm -rf "$OUT" 2>/dev/null || true
mkdir -p "$OUT/model" "$OUT/samples" "$OUT/licenses"
cp "$ROOT"/web/{index.html,app.js,worker.js,serve.py,start.command,vercel.json} "$OUT/"
cp "$BIN/audiocpp_cli.js" "$BIN/audiocpp_cli.wasm" "$OUT/"
cp "$GGUF" "$OUT/model/re-use-lite.gguf"
cp "$ROOT/vendor/noisy_audio/mic_test2.wav" "$OUT/samples/"
cp "$ROOT/vendor/LICENSE" "$OUT/licenses/RE-USE-OpenMDW-1.1.txt"
cp "$TOOLS/ext/audio.cpp/LICENSE" "$OUT/licenses/audio.cpp-Apache-2.0.txt"
cp "$ROOT/release/NOTICE" "$OUT/licenses/NOTICE.txt"
cp "$ROOT/patches/audiocpp-reuse-depth.patch" "$OUT/licenses/"
chmod +x "$OUT/start.command" "$OUT/serve.py" 2>/dev/null || true
find "$OUT" -name '._*' -delete 2>/dev/null || true
echo "web app in $OUT ($(du -sh "$OUT" | cut -f1))"
