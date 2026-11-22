// RE-USE Lite browser page. The heavy lifting is audio.cpp's CLI in worker.js.
"use strict";
const $ = id => document.getElementById(id);
const MODEL_URL = "model/re-use-lite.gguf";
let modelBlob = null, worker = null, urls = [];

function status(msg, cls = "") { const s = $("status"); s.textContent = msg; s.className = cls; }
function log(line) { const l = $("log"); l.textContent += line + "\n"; }

function unsupported() {
  if (location.protocol === "file:")
    return "This page has to be served, not opened as a file. Double-click start.command (macOS) or run: python3 serve.py";
  if (!window.crossOriginIsolated)
    return "The page is not cross-origin isolated (COOP/COEP headers missing), so WebAssembly threads cannot start.";
  if (!navigator.gpu) return "This browser has no WebGPU. Use a current Chrome, Edge or Brave.";
  if (typeof WebAssembly.Suspending !== "function")
    return "This browser lacks WebAssembly JSPI. Use a current Chrome, Edge or Brave.";
  return null;
}

async function getModel() {
  if (modelBlob) return modelBlob;
  status("Downloading the model (about 10 MB, once)…");
  const r = await fetch(MODEL_URL);
  if (!r.ok) throw new Error(`Model download failed: HTTP ${r.status}`);
  modelBlob = await r.blob();
  return modelBlob;
}

function wavRate(buf) {
  const v = new DataView(buf);
  if (buf.byteLength < 28 || v.getUint32(0, false) !== 0x52494646 || v.getUint32(8, false) !== 0x57415645) return 0;
  for (let p = 12; p + 8 <= buf.byteLength;) {
    const id = v.getUint32(p, false), size = v.getUint32(p + 4, true);
    if (id === 0x666d7420) return v.getUint32(p + 12, true); // "fmt "
    p += 8 + size + (size & 1);
  }
  return 0;
}

function encodeWav(samples, rate) { // mono float32 WAV
  const buf = new ArrayBuffer(44 + samples.length * 4), v = new DataView(buf);
  const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 4, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 3, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 4, true); v.setUint16(32, 4, true); v.setUint16(34, 32, true);
  str(36, "data"); v.setUint32(40, samples.length * 4, true);
  new Float32Array(buf, 44).set(samples);
  return buf;
}

async function toWav(blob) {
  const buf = await blob.arrayBuffer();
  const choice = $("rate").value, native = wavRate(buf);
  let rate = choice === "keep" ? native : Number(choice);
  if (!(rate >= 8000 && rate <= 48000)) rate = 24000;
  const ctx = new OfflineAudioContext(1, 1, rate);
  const audio = await ctx.decodeAudioData(buf.slice(0));
  const mono = new Float32Array(audio.length);
  for (let c = 0; c < audio.numberOfChannels; c++) {
    const ch = audio.getChannelData(c);
    for (let i = 0; i < mono.length; i++) mono[i] += ch[i] / audio.numberOfChannels;
  }
  return { wav: encodeWav(mono, rate), seconds: audio.length / rate, rate };
}

function runWorker(model, wavBuf, seconds) {
  return new Promise((resolve, reject) => {
    worker?.terminate();
    worker = new Worker("worker.js");
    const args = ["--task", "s2s", "--family", "reuse", "--model", "/model.gguf", "--backend", "webgpu",
      "--threads", "1", "--audio", "/input.wav", "--out", "/output.wav", "--log"];
    if (seconds > 15) args.push("--request-option", "audio_chunk_duration_sec=10", "--request-option", "audio_chunk_overlap_sec=1");
    worker.onmessage = ({ data }) => {
      if (data.type === "status") status(data.message);
      else if (data.type === "log") log(data.line);
      else if (data.type === "done") { worker.terminate(); worker = null; resolve(data); }
      else if (data.type === "error") { worker.terminate(); worker = null; reject(new Error(data.message)); }
    };
    worker.onerror = e => { reject(new Error(e.message || "worker error")); };
    worker.postMessage({ args, files: [{ path: "/model.gguf", file: model }, { path: "/input.wav", file: new Blob([wavBuf]) }] });
  });
}

function setBusy(b) { for (const id of ["rec", "sample", "rate"]) $(id).disabled = b; $("file").disabled = b; }

async function enhance(blob, name) {
  setBusy(true); $("log").textContent = "";
  try {
    const model = await getModel();
    status("Reading audio…");
    const { wav, seconds, rate } = await toWav(blob);
    const res = await runWorker(model, wav, seconds);
    urls.forEach(URL.revokeObjectURL); urls = [];
    const before = URL.createObjectURL(new Blob([wav], { type: "audio/wav" }));
    const after = URL.createObjectURL(new Blob([res.bytes], { type: "audio/wav" }));
    urls.push(before, after);
    $("before").src = before; $("after").src = after; $("download").href = after;
    $("download").download = (name || "audio").replace(/\.[^.]+$/, "") + "_enhanced.wav";
    $("timing").textContent = `${seconds.toFixed(1)} s at ${rate / 1000} kHz in ${(res.elapsedMs / 1000).toFixed(1)} s`;
    $("results").hidden = false;
    status("Done.", "ok");
  } catch (e) {
    status(e.message.split("\n")[0], "err"); log(e.stack || e.message);
  } finally { setBusy(false); }
}

// Test hook (used by web_check.mjs): enhance a WAV exactly as given, no decoding or resampling.
window.reuseLite = {
  async enhanceWav(arrayBuffer) {
    const res = await runWorker(await getModel(), arrayBuffer, 0);
    return Array.from(res.bytes);
  },
};

let recorder = null;
$("rec").onclick = async () => {
  if (recorder) { recorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { noiseSuppression: false, echoCancellation: false, autoGainControl: false } });
    const chunks = []; recorder = new MediaRecorder(stream);
    recorder.ondataavailable = e => chunks.push(e.data);
    recorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop()); recorder = null; $("rec").textContent = "Record";
      enhance(new Blob(chunks, { type: chunks[0]?.type || "audio/webm" }), "recording");
    };
    recorder.start(); $("rec").textContent = "Stop"; $("rec").disabled = false; status("Recording… press Stop when done.");
  } catch (e) { status("Microphone not available: " + e.message, "err"); }
};
$("file").onchange = () => { const f = $("file").files[0]; if (f) enhance(f, f.name); $("file").value = ""; };
$("sample").onclick = async () => {
  const r = await fetch("samples/mic_test2.wav");
  enhance(await r.blob(), "mic_test2.wav");
};

const why = unsupported();
if (why) { $("unsupported").textContent = why; $("unsupported").hidden = false; setBusy(true); status("Not supported here.", "err"); }
