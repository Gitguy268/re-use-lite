// Drive the built web app in Playwright Chromium with a real WebGPU adapter and save its outputs.
// usage: node web_check.mjs <web_dir> <inputs_dir> <out_dir>
//        node web_check.mjs https://live.url/     (live site: only the "Try the sample" UI test)
// Serves <web_dir> (with COOP/COEP) plus /check-inputs/ from <inputs_dir>, then for every WAV calls
// window.reuseLite.enhanceWav (the exact WAV, no decoding) and writes the result to <out_dir>.
import { createRequire } from "module";
import fs from "fs";
import http from "http";
import path from "path";
const require = createRequire(process.env.PW_DIR ? path.join(process.env.PW_DIR, "x.js") : import.meta.url);
const { chromium } = require("playwright");

const live = /^https?:\/\//.test(process.argv[2] || "");
const [webDir, inDir, outDir] = live ? [] : process.argv.slice(2).map(p => path.resolve(p));
const types = { ".html": "text/html", ".js": "text/javascript", ".wasm": "application/wasm", ".wav": "audio/wav", ".json": "application/json" };
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const file = u.startsWith("/check-inputs/") ? path.join(inDir, u.slice(14)) : path.join(webDir, u === "/" ? "index.html" : u);
  if (!file.startsWith(webDir) && !file.startsWith(inDir)) { res.writeHead(403); return res.end(); }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "Content-Type": types[path.extname(file)] || "application/octet-stream",
      "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Resource-Policy": "same-origin" });
    res.end(data);
  });
});
if (!live) await new Promise(r => server.listen(0, "127.0.0.1", r));
const url = live ? process.argv[2] : `http://127.0.0.1:${server.address().port}/`;
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=metal"] });
const page = await browser.newPage();
const logs = [];
page.on("console", m => logs.push(m.text()));
page.on("pageerror", e => logs.push("pageerror: " + e.message));
await page.goto(url, { waitUntil: "load", timeout: 120000 });
const unsupported = await page.evaluate(() => document.getElementById("unsupported").hidden ? null : document.getElementById("unsupported").textContent);
if (unsupported) { console.error("page says unsupported:", unsupported); process.exit(2); }
let n = 0;
if (!live) fs.mkdirSync(outDir, { recursive: true });
for (const f of live ? [] : fs.readdirSync(inDir).filter(f => f.endsWith(".wav") && !f.startsWith("._")).sort()) {
  const t0 = Date.now();
  const bytes = await page.evaluate(async name => {
    const buf = await (await fetch("/check-inputs/" + name)).arrayBuffer();
    return await window.reuseLite.enhanceWav(buf);
  }, f);
  fs.writeFileSync(path.join(outDir, f), Buffer.from(bytes));
  console.log(`browser: ${f} ${(Date.now() - t0) / 1000}s`); n++;
}
// Long recording through the real file picker (> 15 s takes the chunked path), local mode only
function readWavF32(file) { // mono float32 WAV as written by soundfile/check_formats
  const b = fs.readFileSync(file); let p = 12, rate = 0;
  while (p + 8 <= b.length) {
    const id = b.toString("ascii", p, p + 4), size = b.readUInt32LE(p + 4);
    if (id === "fmt ") rate = b.readUInt32LE(p + 12);
    if (id === "data") return { rate, data: new Float32Array(b.buffer.slice(b.byteOffset + p + 8, b.byteOffset + p + 8 + size)) };
    p += 8 + size + (size & 1);
  }
  throw new Error("no data chunk in " + file);
}
function writeWavF32(file, data, rate) {
  const h = Buffer.alloc(44);
  h.write("RIFF", 0); h.writeUInt32LE(36 + data.length * 4, 4); h.write("WAVE", 8); h.write("fmt ", 12);
  h.writeUInt32LE(16, 16); h.writeUInt16LE(3, 20); h.writeUInt16LE(1, 22); h.writeUInt32LE(rate, 24);
  h.writeUInt32LE(rate * 4, 28); h.writeUInt16LE(4, 32); h.writeUInt16LE(32, 34); h.write("data", 36); h.writeUInt32LE(data.length * 4, 40);
  fs.writeFileSync(file, Buffer.concat([h, Buffer.from(data.buffer)]));
}
if (!live) {
  const parts = fs.readdirSync(inDir).filter(f => f.endsWith("_16k.wav") && !f.startsWith("._")).sort()
    .map(f => readWavF32(path.join(inDir, f)).data);
  let total = [], n16 = 0;
  while (n16 < 50 * 16000) for (const p of parts) { total.push(p); n16 += p.length; }
  const long = new Float32Array(n16); let o = 0; for (const p of total) { long.set(p, o); o += p.length; }
  const longPath = path.join(outDir, "_long_16k_input.wav");
  writeWavF32(longPath, long, 16000);
  await page.selectOption("#rate", "16000");
  const t0 = Date.now();
  await page.setInputFiles("#file", longPath);
  await page.waitForFunction(() => /Done|Model|fail|error|exited/i.test(document.getElementById("status").textContent) &&
    !/Downloading|Reading|Starting|Enhancing/.test(document.getElementById("status").textContent), null, { timeout: 600000 });
  const r = await page.evaluate(async () => {
    const st = document.getElementById("status").textContent;
    const src = document.getElementById("after").src;
    if (!/Done/.test(st)) return { st };
    const b = await (await fetch(src)).arrayBuffer(); const v = new DataView(b); let p = 12;
    while (p + 8 <= b.byteLength) {
      const id = String.fromCharCode(v.getUint8(p), v.getUint8(p + 1), v.getUint8(p + 2), v.getUint8(p + 3)), size = v.getUint32(p + 4, true);
      if (id === "data") {
        const bits = v.getUint16(34, true), n = size / (bits / 8);
        let sum = 0, finite = true;
        for (let i = 0; i < n; i++) {
          const x = bits === 32 ? v.getFloat32(p + 8 + i * 4, true) : v.getInt16(p + 8 + i * 2, true) / 32768;
          if (!Number.isFinite(x)) finite = false; sum += x * x;
        }
        return { st, n, finite, rms: Math.sqrt(sum / n), timing: document.getElementById("timing").textContent };
      }
      p += 8 + size + (size & 1);
    }
    return { st, err: "no data chunk" };
  });
  console.log(`long file: ${long.length / 16000}s in ${(Date.now() - t0) / 1000}s ->`, JSON.stringify(r));
  if (!(r.n === long.length && r.finite && r.rms > 1e-4)) { console.error("long-file check FAILED"); console.error(logs.slice(-30).join("\n")); process.exit(1); }
  console.log("long file: OK");
}
// UI path once: the sample button end to end
await page.evaluate(() => { document.getElementById("status").textContent = ""; });
await page.click("#sample");
await page.waitForFunction(() => /Done|error|fail/i.test(document.getElementById("status").textContent), null, { timeout: 300000 });
const ui = await page.evaluate(() => ({ status: document.getElementById("status").textContent, timing: document.getElementById("timing").textContent }));
console.log("ui:", JSON.stringify(ui));
await browser.close(); if (!live) server.close();
if (!/Done/.test(ui.status)) { console.error(logs.slice(-30).join("\n")); process.exit(1); }
console.log(`browser: ${n} files OK`);
