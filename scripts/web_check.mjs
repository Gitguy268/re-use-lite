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
// UI path once: the sample button end to end
await page.click("#sample");
await page.waitForFunction(() => /Done|error|fail/i.test(document.getElementById("status").textContent), null, { timeout: 300000 });
const ui = await page.evaluate(() => ({ status: document.getElementById("status").textContent, timing: document.getElementById("timing").textContent }));
console.log("ui:", JSON.stringify(ui));
await browser.close(); if (!live) server.close();
if (!/Done/.test(ui.status)) { console.error(logs.slice(-30).join("\n")); process.exit(1); }
console.log(`browser: ${n} files OK`);
