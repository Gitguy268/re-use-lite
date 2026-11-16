"""Gate 0b: local port (CPU + MPS) vs the official Space on vendor/noisy_audio/mic_test2.wav."""
import os, sys, time, json, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, soundfile as sf, numpy as np
from reuse_lite.model import load_teacher, enhance, BACKEND, VENDOR, ROOT

OUT = os.path.join(ROOT, "results", "parity"); os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(VENDOR, "noisy_audio", "mic_test2.wav")


def si_sdr(est, ref):
    n = min(len(est), len(ref)); est, ref = est[:n].astype(np.float64), ref[:n].astype(np.float64)
    a = np.dot(est, ref) / np.dot(ref, ref); t = a * ref
    return 10 * np.log10(np.sum(t ** 2) / np.sum((est - t) ** 2))


wav, sr = sf.read(SRC, dtype="float32")
if wav.ndim > 1: wav = wav.mean(1)
print("backend", BACKEND, "sr", sr, "dur", len(wav) / sr)
outs = {}
for dev in ["cpu", "mps"]:
    m, cfg = load_teacher(dev)
    x = torch.from_numpy(wav)[None].to(dev)
    t0 = time.time(); y = enhance(m, cfg, x, sr);
    if dev == "mps": torch.mps.synchronize()
    dt = time.time() - t0
    outs[dev] = y[0].cpu().numpy()
    sf.write(os.path.join(OUT, f"port_{dev}.wav"), outs[dev], sr)
    print(f"{dev}: {dt:.2f}s  RTF={dt / (len(wav) / sr):.3f}")
res = {"cpu_vs_mps_sisdr": si_sdr(outs["mps"], outs["cpu"]),
       "port_vs_noisy_sisdr": si_sdr(outs["cpu"], wav)}

if "--space" in sys.argv:
    from gradio_client import Client, handle_file
    c = Client("nvidia/NVIDIA-NemotronLabs-RE-USE", token=__import__("huggingface_hub").get_token())
    r = c.predict(handle_file(SRC), None, "", "", api_name="/unified_enhance")
    enh = r[1] if isinstance(r, (list, tuple)) else r
    enh = enh if isinstance(enh, str) else enh.get("path", enh)
    shutil.copy(enh, os.path.join(OUT, "space.wav"))
    sp, ssr = sf.read(enh, dtype="float32")
    if sp.ndim > 1: sp = sp.mean(1)
    assert ssr == sr, (ssr, sr)
    res["port_cpu_vs_space_sisdr"] = si_sdr(outs["cpu"], sp)
    res["port_mps_vs_space_sisdr"] = si_sdr(outs["mps"], sp)
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(OUT, "parity.json"), "w"), indent=1)
