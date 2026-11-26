# Code & Pipeline Review Notes for Claude / Fable Agents

> Strictly observational notes per `HANDOFF.md` coordination rules (no code files edited by Gemini).

---

## 1. `ai_gen.py` — SD-Turbo vs. SD 1.5 LoRA Compatibility (High Impact)
- `ai_gen.py` loads `stabilityai/sd-turbo`, which is distilled from **SD 2.1** (1024-dim OpenCLIP ViT-H/14 text encoder), **not** SD 1.5 (768-dim CLIP ViT-L/14).
- Line 9 & 44 (`--lora path/to/dmt_lora.safetensors` → `pipe.load_lora_weights(lora)`) will fail with a weight-shape mismatch if you later trains an **SD 1.5** LoRA.
- **Suggestion:** Either document `--lora` as requiring an **SD 2.1 / SD-Turbo LoRA**, or add a `--base sd15-lcm` option loading `stable-diffusion-v1-5/stable-diffusion-v1-5` + `latent-consistency/lcm-lora-sdv1-5` (2–4 steps, TAESD, circular padding) so standard SD 1.5 LoRAs plug straight in.
- **GTX 1650 Ti `--fp32` VRAM note:** Full FP32 `sd-turbo` UNet weights (~3.4 GB) + TAESD + activations may OOM on 4 GB VRAM without `pipe.enable_model_cpu_offload()`. If `--fp16` produces NaN/black frames on TU117, testing **FP16 UNet + FP32 TAESD** first (before full `--fp32`) keeps VRAM around ~2.3 GB while fixing VAE-induced NaNs.

---

## 2. ThinkPad Hybrid GPU (Intel UHD + GTX 1650 Ti) — Spout & ModernGL
- On Windows Optimus laptops, `.venv\Scripts\python.exe` defaults to the **Intel UHD iGPU** unless pinned to **High Performance (NVIDIA)** in *Windows Settings → System → Display → Graphics*.
- If Python renders on Intel while TouchDesigner/Resolume runs on NVIDIA, **Spout2 texture sharing silently fails (black feed)** and complex Fable GLSL shaders will bottleneck on Intel UHD.
- **Suggestion:** Check `[gl] ...` printed at `live_visuals.py` startup (verify it prints `NVIDIA GeForce GTX 1650 Ti...`, not `Intel(R) UHD Graphics`) and document the Windows Graphics Settings pinning step in `README.md` / `HANDOFF.md`.

---

## 3. `live_visuals.py` — Notes for the Fable 5.2 Shader Upgrade Pass
- **AI texture UV utilization (line 67):** `vec2 tuv = vec2(log(r+1e-3)*0.5 - ..., an*0.5);` only samples half (`0.0..0.5`) of the AI texture horizontally, and `log(r+1e-3)` pinches tightly at `r=0`. Mapping `an` across `0.0..1.0` (with mirrored repeat or seamless circular texture) doubles visible AI detail; adding a gentle smoothstep mask or golden-spiral core glow at `r < 0.04` hides the log-polar pole cleanly.
- **Beadwork ring transitions (lines 75–82):** `nb = max(1.0, floor(r*s*0.5/ringW))` changes discontinuously with `r`, which is fine statically, though `floor(aa)` in hue calculation can subtly flicker on beads right at a radius step. Using continuous polar angle `an` (or `sin(ring * ... + an * ...)`) for color variation eliminates any ring-seam flicker while keeping jewel-tone spectral cycling crisp.
- **SpoutGL vertical orientation (line 337):** `spout.sendTexture(pp[dst].glo, 0x0DE1, rw, rh, True, 0)` passes `bInvert=True` — worth verifying orientation once TouchDesigner `Syphon Spout In TOP` or Resolume receives `PsyProjector`.
- **Visual-quality ideas inspired by TAS / Android Jones / Jonathan Singer / Gorge:**
  - Replace single-octave value noise (`noise()`) with 2–3 octave **curl / domain-warped FBM** for organic "swimming" fluid motion (very cheap at 1280×800 on GTX 1650 Ti — easily stays >120 FPS uncapped).
  - Add **Shipibo / sacred-geometry SDF filigree layer** (interlocking stepped lines, flower-of-life arcs, or hyperbolic Poincaré disk reflections) composited cleanly with additive/screen bloom on pure `#000000` black.
  - Add **spectral palette presets** (key `P`): e.g. *Aya Jewel* (turquoise/magenta/gold/indigo), *Alex Grey Cosmic* (fiery gold/violet/cyan), *Gorge Twilight* (deep sapphire/amber/electric teal), *Emerald Shipibo*.
  - Add **4-bar / slow-breathing LFO modulation** alongside fast audio transients so breakdowns stay hypnotic instead of freezing flat when bass drops out.
