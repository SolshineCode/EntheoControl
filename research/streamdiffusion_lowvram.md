# StreamDiffusion / StreamDiffusionTD / StreamTurbo on 4 GB GTX 1650 Ti (Windows, 2026)

> Target GPU: **NVIDIA GeForce GTX 1650 Ti with Max-Q Design** — **4 GB VRAM**, Turing TU117 (**compute capability 7.5**), driver **581.57** (CUDA 13 capable), **no Tensor Cores**, 35 W Max-Q power envelope.

---

## 1. Current (2026) Ecosystem & Best Option for 4 GB Windows

| Project | Repo / Site | Status in 2026 | Fit for 4 GB GTX 1650 Ti |
|---|---|---|---|
| **StreamTurbo** (`tedliou/StreamTurbo`) | https://github.com/tedliou/StreamTurbo | Active (v1.0.0, Windows-only standalone rewrite of StreamDiffusion for TD over Spout/NDI) | **Best live-AI candidate** — automatic **Low-VRAM Mode** (4–6 GB), single-step `sd-turbo` + tiny VAE, one-click `start.cmd`, driver 580+ supported (we have 581.57). |
| **StreamDiffusion** (`cumulo-autumn/StreamDiffusion`) | https://github.com/cumulo-autumn/StreamDiffusion | Original upstream repo; manual Python/CUDA/TensorRT dependency friction on Windows | Usable via Windows forks (`sdbds/StreamDiffusion-for-windows`), but fragile to set up compared to StreamTurbo. |
| **StreamDiffusionTD** (DotSimulate) | https://www.dotsimulate.com/ | Modular TouchDesigner `.tox` ecosystem (supports local TRT + RunPod cloud fallback) | Local mode is tight on 4 GB VRAM alongside TD; cloud RunPod fallback is viable if you ever want 30+ FPS SDXL/Flux on gig Wi-Fi. |
| **Async Hybrid (`ai_gen.py` → `live_visuals.py` / TD)** | Local (`Documents\Projection`) | Works today, zero mid-set OOM/thermal risk | **Best for live EDM shows** — locked 60 FPS shaders at 1280×800 over pre-generated seamless AI textures. |

---

## 2. Real Reported FPS & Realistic GTX 1650 Ti Expectations

### Published 4 GB Benchmark (from `tedliou/StreamTurbo` README)
- **Hardware:** NVIDIA **RTX 3050 Laptop 4 GB** (Ampere, *has Tensor Cores*), Low-VRAM Mode, TensorRT engine at `512 × 384`:
  - With AI Upscaler (`→ 1024 × 768`) + Temporal Coherence ON: **p50 = 163 ms (~5.8 FPS)**
  - With Upscaler OFF + Temporal Coherence OFF: **p50 = 116 ms (~8.2 FPS)**

### Adjusting for GTX 1650 Ti Max-Q (Turing TU117, 4 GB, No Tensor Cores)
- Unlike the RTX 20/30 series, the **GTX 1650 Ti** lacks hardware Tensor Cores and runs at a 35 W Max-Q thermal limit.
- **Realistic expectation:** **~1.5–4.0 FPS** at `512 × 384` or `512 × 512` (1-step `sd-turbo` + TAESD) with TensorRT, and **~0.8–2.0 FPS** in raw PyTorch.
- **Practical takeaway for live projection:** Even 2–3 FPS of AI morphing looks hypnotically smooth **if** you feed it via Spout into a **60 FPS TouchDesigner / `live_visuals.py` feedback + crossfade + warp shader** so the AI slowly mutates the source texture while the shader drives 60 FPS kaleidoscopic motion and audio-reactive pulses. Just watch GPU thermals and VRAM headroom (~3.2–3.6 GB used by AI leaves very little for heavy TD networks simultaneously).

---

## 3. GTX 16xx (Turing TU117) `fp16` NaN / Black-Image Issue

GTX 1650 / 1650 Ti / 1660 cards are notorious for producing **solid black frames (`NaN` tensor overflow)** when running Stable Diffusion in `torch.float16`:
- **Root cause:** Turing GTX 16-series lacks RTX Tensor Core FP16 accumulation paths; intermediate activations in the standard SD 1.5/2.1 `AutoencoderKL` VAE (and occasionally attention layers in the UNet) overflow FP16's max range (`65504`), turning the entire decoded image to `NaN` → black pixels.
- **Proven fixes (ordered fastest → safest):**
  1. **Swap VAE for TAESD (`madebyollin/taesd`):** Much smaller dynamic range and ~10× faster decoding — usually avoids VAE FP16 NaN while cutting VRAM by ~300–500 MB. *(Already implemented in `ai_gen.py`!)*
  2. **Upcast VAE only (`--no-half-vae` / `vae` in `torch.float32`, UNet in `torch.float16`):** Fixes ~95% of GTX 16xx black-image cases with minimal VRAM penalty (~200 MB).
  3. **Full `torch.float32` (`--fp32`):** Guaranteed NaN-free if the UNet itself overflows on a specific prompt/seed, at ~2× VRAM & slower inference. Note: full FP32 `sd-turbo` UNet (~3.4 GB weights alone) is very tight on 4 GB VRAM — pair with `pipe.enable_model_cpu_offload()` or `pipe.enable_sequential_cpu_offload()` if `--fp32` OOMs.
  4. **PyTorch SDPA / xFormers:** PyTorch 2.x built-in Scaled Dot-Product Attention (`SDPA`) + `enable_attention_slicing()` keeps attention peaks small and generally behaves better numerically on Turing than naive unchunked attention.

---

## 4. Does TensorRT Work on Turing (SM 7.5) with 4 GB VRAM?

- **Architecture support:** **Yes.** NVIDIA TensorRT supports Turing compute capability **7.5** (which GTX 1650 Ti reports via `nvidia-smi`: `compute_cap = 7.5`). Because TU117 has no Tensor Cores, TensorRT uses optimized CUDA FP16/FP32 fused kernels rather than Tensor Core HMMA kernels — still noticeably faster and lower-VRAM than eager PyTorch, just a smaller speedup multiplier than on RTX cards.
- **4 GB VRAM build constraints:** Building a TensorRT engine normally spikes VRAM above 4 GB if PyTorch holds the model at the same time.
  - **How StreamTurbo solves it:** Its automatic **Low-VRAM Mode** (triggered whenever VRAM < 6 GB) temporarily pauses PyTorch output and releases VRAM so TensorRT has exclusive GPU memory during the ~4-minute first-time engine build, and restricts the dynamic resolution profile to **`256–512 px`** per side.
  - **Gotcha to test:** StreamTurbo's README lists "NVIDIA RTX 20 series (Turing) or newer, 4 GB+ VRAM, driver 580+". Since GTX 1650 Ti is Turing SM 7.5 with driver 581.57, `streamturbo doctor` should pass the SM/driver/VRAM checks, **but** watch for FP16 NaN outputs inside the built TensorRT engine on GTX 16xx. If TRT outputs black frames, StreamTurbo automatically falls back to PyTorch (or can be run in PyTorch mode).

---

## 5. Exact Install Steps for StreamTurbo (Windows)

StreamTurbo is self-contained inside its own folder (bundled `uv`, Python 3.12, CUDA runtime, models, and TRT engines — delete the folder to uninstall cleanly):

```bat
cd C:\path\to\EntheoControl
git clone --branch v1.0.0 https://github.com/tedliou/StreamTurbo.git
cd StreamTurbo
start.cmd
```

What `start.cmd` does automatically:
1. Verifies NVIDIA driver ≥ 580 (this laptop has `581.57` — passes).
2. Downloads pinned `uv` into `cache\bin` and syncs Python 3.12 + PyTorch/CUDA/TensorRT packages (~several GB).
3. Runs `.venv\Scripts\streamturbo doctor` (reports GPU, compute cap, VRAM, Low-VRAM Mode trigger, NDI/Spout availability).
4. Downloads `sd-turbo` + tiny VAE weights (~2.3 GB into `models\`) and opens the Web Panel (`http://localhost:8765`).

Useful commands before a show:
```bat
REM Verify GPU, Spout, and Low-VRAM detection
.venv\Scripts\streamturbo doctor

REM Pre-build TensorRT engines before a gig (so no 4-min pause live)
.venv\Scripts\streamturbo build-engines --low-vram

REM List active Spout / NDI senders (e.g., PsyProjector or TouchDesigner)
.venv\Scripts\streamturbo run --list-sources

REM Run using Spout I/O instead of default NDI
.venv\Scripts\streamturbo run --settings spout.toml --low-vram
```

---

## Sources
- StreamTurbo GitHub repository & README (Low-VRAM Mode, RTX 3050 4 GB benchmark, requirements): https://github.com/tedliou/StreamTurbo
- StreamTurbo Setup & 4 GB GPU Guide (vervecode.dev): https://vervecode.dev/ai/streamturbo/
- StreamDiffusion upstream repository: https://github.com/cumulo-autumn/StreamDiffusion
- StreamDiffusionTD (DotSimulate): https://www.dotsimulate.com/
- Diffusers TAESD (`madebyollin/taesd`) model card: https://huggingface.co/madebyollin/taesd
- Stability AI `sd-turbo` model card: https://huggingface.co/stabilityai/sd-turbo
