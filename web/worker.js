// SPDX-License-Identifier: Apache-2.0
// Adapted from audio.cpp examples/webgpu/worker.js (Copyright 2026 ShugoAI LLC, Apache-2.0).
// Runs the audio.cpp CLI (WebAssembly + WebGPU) in a worker: writes the model and input into the
// in-memory file system, calls main, and returns the output WAV.
importScripts("audiocpp_cli.js");

self.onmessage = async ({ data }) => {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("This browser did not provide a WebGPU adapter.");
    if (!adapter.features.has("shader-f16")) throw new Error("This GPU/browser lacks the WebGPU shader-f16 feature.");
    self.postMessage({ type: "status", message: "Starting WebAssembly…" });
    const module = await createAudioCpp({
      mainScriptUrlOrBlob: new URL("audiocpp_cli.js", self.location.href).href,
      print: line => self.postMessage({ type: "log", line }),
      printErr: line => self.postMessage({ type: "log", line }),
    });
    for (const { path, file } of data.files) {
      const stream = module.FS.open(path, "w");
      try {
        for (let offset = 0; offset < file.size; offset += 16 * 1024 * 1024) {
          const bytes = new Uint8Array(await file.slice(offset, offset + 16 * 1024 * 1024).arrayBuffer());
          module.FS.write(stream, bytes, 0, bytes.length, offset);
        }
      } finally {
        module.FS.close(stream);
      }
    }
    self.postMessage({ type: "status", message: "Enhancing on your GPU…" });
    const started = performance.now();
    const status = await module.callMain(data.args);
    if (status !== 0) throw new Error(`audio.cpp exited with status ${status}. See the log.`);
    const bytes = module.FS.readFile("/output.wav");
    self.postMessage({ type: "done", elapsedMs: performance.now() - started, bytes }, [bytes.buffer]);
  } catch (error) {
    self.postMessage({ type: "error", message: error.stack || error.message || String(error) });
  }
};
