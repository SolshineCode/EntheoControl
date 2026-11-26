# Live Psychedelic Projection-Mapping VJ Engine — Research Support Brief

**Target Hardware & Stack:** Windows laptop (ThinkPad Optimus, Intel Core i9-10885H, NVIDIA GeForce GTX 1650 Ti Max-Q, 4 GB GDDR6 VRAM, Turing TU117, Compute Capability 7.5, fp16 supported at 2:1 rate, **no Tensor Cores**, **no hardware bf16**), Python 3.11, Hugging Face `diffusers`. ([TechPowerUp](https://www.techpowerup.com/gpu-specs/geforce-gtx-1650-ti-max-q.c3582), [NVIDIA Turing Architecture](https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/))

---

## Recommendations for the Team (10-Line Summary)

1. **Primary local fast model (4 GB VRAM):** Start with **`IDKiro/sdxs-512-dreamshaper`** (315.7M params, 1 step, `guidance_scale=0`, `fp16`, `openrail++` commercial-OK) for ultra-low VRAM (~1.5–2.5 GB) alongside OpenGL; skip the older `IDKiro/sdxs-512-0.9`, and note that **SDXS-1024 was never released**. ([HF SDXS-DreamShaper](https://huggingface.co/IDKiro/sdxs-512-dreamshaper), [HF SDXS-0.9](https://huggingface.co/IDKiro/sdxs-512-0.9))
2. **Best local recognizable-subject quality in 4 GB:** Pair **`ByteDance/Hyper-SD`** (`Hyper-SD15-1step-lora` with `TCDScheduler` or `Hyper-SD15-2steps-lora` / `4steps-lora` with `DDIMScheduler(timestep_spacing="trailing")`, CreativeML Open RAIL++-M) with a strong SD 1.5 checkpoint (e.g., `Lykon/dreamshaper-8`) + **`madebyollin/taesd`** (MIT) for portraits, statues, and animals. ([HF Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD), [Hyper-SD License](https://huggingface.co/ByteDance/Hyper-SD/blob/main/LICENSE.md))
3. **Avoid PixArt-Sigma and SANA-0.6B/Sprint on this laptop:** Both look small (~600M DiT) but fail a live 4 GB Turing budget — PixArt uses a 4.7B T5-XXL text encoder (~9.5 GB fp16) and SANA uses Gemma-2-2B (~5.2 GB fp16) plus **native BF16** weights that overflow to NaN/black images in pure fp16 on Turing. ([HF Sana-Sprint](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers), [HF PixArt-LCM](https://huggingface.co/PixArt-alpha/PixArt-LCM-XL-2-1024-MS))
4. **SD-Turbo commercial live shows:** `stabilityai/sd-turbo` is **free for commercial use if your aggregate annual gross revenue is under USD $1,000,000**, *provided you register* at `https://stability.ai/community-license` and follow AUP/attribution rules; ≥ $1M requires an Enterprise License. ([SD-Turbo LICENSE.md](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md))
5. **Best realistic Turing (TU117) speedups:** Swap the VAE for **`madebyollin/taesd`**, use PyTorch 2.x native **SDPA** (`AttnProcessor2_0`), `torch.float16`, and **`torch.channels_last`**; avoid StreamDiffusion multi-step `Stream Batch` (multiplies VRAM → OOM on 4 GB) and treat Windows `torch.compile` / TensorRT as optional benchmarks because TU117 lacks Tensor Cores. ([HF TAESD](https://huggingface.co/madebyollin/taesd), [Diffusers FP16/Compile Docs](https://huggingface.co/docs/diffusers/main/en/optimization/fp16))
6. **Cloud Hero-Frame Generation (Nano Banana):** Prefer calling the **paid Gemini Developer API directly** (`gemini-3.1-flash-lite-image` = Nano Banana 2 Lite for **sub-2s** 1K @ $0.0336/img, or `gemini-nano-banana-2.1` for best quality/multi-ref @ $0.0336/img) over wrapping `agy -p` in a commercial VJ engine, because **Antigravity Terms prohibit third-party software/tools accessing the Antigravity Service via Antigravity OAuth**. ([Gemini Image Gen Docs](https://ai.google.dev/gemini-api/docs/image-generation), [Antigravity Terms](https://antigravity.google/terms))
7. **If scripting `agy` CLI personally/non-interactively:** Use `agy -p "<prompt naming input/output PNG paths>" --model gemini-4-hydrogen-low --effort low --mode=accept-edits --output-format json --print-timeout 120s` (or persistent `--input-format stream-json --output-format stream-json`); note `--model` picks the reasoning agent, while Antigravity's built-in image tool is fixed to **Nano Banana 2** and image inputs must be passed as file paths (not `stream-json` image blocks). ([Antigravity Headless Docs](https://antigravity.google/docs/cli/headless), [Antigravity Models](https://antigravity.google/docs/models))
8. **Hard photosensitive safety guardrails:** Enforce **≤ 2.5 Hz** strobe cap (strictly **≤ 3 flashes / 6 opposing transitions in any rolling 1.0 s window**), minimum **350–500 ms** hold/fade on blackouts, clamp oscillating full-frame luminance swings (`ΔL < 0.08` above 2.5 Hz), **never strobe saturated red** (`R/(R+G+B) ≥ 0.75`), and limit high-contrast repeating stripes/checkerboards to ≤ 5 pairs. ([WCAG 2.2 SC 2.3.1](https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html), [ITU-R BT.1702](https://www.itu.int/rec/R-REC-BT.1702))
9. **Laptop thermal control during live sets:** Disable Intel Turbo Boost on the i9-10885H (via `PERFBOOSTMODE = Disabled` or Windows **Maximum Processor State = 99%**) so CPU heat doesn't choke the shared CPU+GPU heatsink, cap OpenGL FPS via **NVIDIA Max Frame Rate** (60 or 30 FPS), stay plugged into full-wattage AC, and **disable WhisperMode** so NVIDIA doesn't dynamically throttle mid-show. ([Notebookcheck X1 Extreme Gen 3](https://www.notebookcheck.net/ThinkPad-X1-Extreme-Gen3-2020-Lenovo-s-Premium-Multimedia-Laptop-with-GTX-1650-Ti-Max-Q-in-Review.513512.0.html), [NVIDIA 441.87 Release Notes](http://us.download.nvidia.com/Windows/441.87/441.87-win10-win8-win7-release-notes.pdf))
10. **Stop NVIDIA OpenGL VSync CPU busy-spin:** Set **Threaded optimization = `Off`** for `python.exe` in NVIDIA Control Panel, keep OpenGL context + `SwapBuffers` on one thread (never call `glFinish()`), and pace frames with an explicit OS sleep (or `DwmFlush()`) before `SwapBuffers` so `nvoglv64.dll` does not peg an i9 core at 100%. ([NVIDIA Dev Forums](https://forums.developer.nvidia.com/t/opengl-vsync-swapbuffers-100-cpu-core/29086), [Win32 DwmFlush Docs](https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/nf-dwmapi-dwmflush))

---

## 1. Small Fast Image Models for 4 GB VRAM (Priority)

### 1.1 Hardware Reality Check: GTX 1650 Ti Max-Q (4 GB, Turing TU117)

Understanding the exact GPU silicon prevents wasted engineering time on optimizations designed for RTX cards:
- **Silicon & Compute Capability:** NVIDIA **TU117** (Turing, Compute Capability **7.5**), 1,024 CUDA cores, 4 GB GDDR6 on a 128-bit bus, **35 W TGP** (Max-Q). ([TechPowerUp GTX 1650 Ti Max-Q](https://www.techpowerup.com/gpu-specs/geforce-gtx-1650-ti-max-q.c3582))
- **No Tensor Cores / No Hardware BF16:** All GTX 16-series cards omit Tensor Cores and RT Cores; hardware `bfloat16` requires Ampere (SM 8.0+) or newer. However, Turing's FP16 CUDA pipelines still deliver **2:1 FP16 throughput** relative to FP32, so `torch.float16` gives both a 2× memory reduction and real compute speedup. ([TechPowerUp](https://www.techpowerup.com/gpu-specs/geforce-gtx-1650-ti-max-q.c3582), [NVIDIA Turing Architecture](https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/))
- **Practical VRAM Budget for a Live VJ Engine:** Windows DWM + an active OpenGL projection-mapping renderer + projector framebuffers/FBOs typically consume **0.4–1.0 GB** of VRAM (*exact engine footprint unverified*). That leaves only **~2.5–3.2 GB** of usable VRAM for the diffusion model + activations before Windows WDDM silently pages tensors to system RAM and causes severe frame drops.

---

### 1.2 SDXS-512 (`IDKiro/sdxs-512-*`) Deep Dive

#### Exact Hugging Face Repository IDs
Querying the Hugging Face Hub API for author `IDKiro` confirms **exactly four public SDXS models** (all 512×512; released March–April 2024): ([HF API `author=IDKiro`](https://huggingface.co/api/models?author=IDKiro), [GitHub `IDKiro/sdxs`](https://github.com/IDKiro/sdxs))
1. **`IDKiro/sdxs-512-0.9`** — Early research checkpoint (328,177,284 total safetensors params; teacher: `stabilityai/sd-turbo`, offline DM: `stabilityai/stable-diffusion-2-1-base`, VAE: `madebyollin/taesd`). The README explicitly states: **"Use the new version for community: `IDKiro/sdxs-512-dreamshaper`. It has better quality and is faster."** ([HF `sdxs-512-0.9`](https://huggingface.co/IDKiro/sdxs-512-0.9), [HF API `sdxs-512-0.9`](https://huggingface.co/api/models/IDKiro/sdxs-512-0.9))
2. **`IDKiro/sdxs-512-dreamshaper`** — Recommended community 512×512 checkpoint (315,746,244 total safetensors params; teacher: `Lykon/dreamshaper-8-lcm`, offline DM: `Lykon/dreamshaper-8`, VAE: `madebyollin/taesd`). ([HF `sdxs-512-dreamshaper`](https://huggingface.co/IDKiro/sdxs-512-dreamshaper), [HF API `sdxs-512-dreamshaper`](https://huggingface.co/api/models/IDKiro/sdxs-512-dreamshaper))
3. **`IDKiro/sdxs-512-dreamshaper-sketch`** — ControlNet sketch-to-image variant for image-conditioned live generation. ([HF `sdxs-512-dreamshaper-sketch`](https://huggingface.co/IDKiro/sdxs-512-dreamshaper-sketch))
4. **`IDKiro/SDXS-512-DreamShaper-Anime`** — Anime-styled LoRA/checkpoint variant. ([HF `SDXS-512-DreamShaper-Anime`](https://huggingface.co/IDKiro/SDXS-512-DreamShaper-Anime))

- **Status of SDXS-1024 and SDXS-512-1.0:** **Never released.** Both model cards state: *"In order to avoid some possible commercial and copyright risks, the SDXS-512-1.0 and SDXS-1024-1.0 will not be available shortly."* The GitHub README adds: *"Unfortunately, the SDXS training code is not allowed to be open-sourced and will most likely not be updated again."* ([HF `sdxs-512-0.9`](https://huggingface.co/IDKiro/sdxs-512-0.9), [GitHub `IDKiro/sdxs`](https://github.com/IDKiro/sdxs))

#### License & Commercial Use
- **Declared License:** All four repositories declare `license: openrail++` (CreativeML Open RAIL++-M) in their Hugging Face YAML metadata. ([HF `sdxs-512-dreamshaper`](https://huggingface.co/IDKiro/sdxs-512-dreamshaper), [HF `sdxs-512-0.9`](https://huggingface.co/IDKiro/sdxs-512-0.9))
- **Commercial Use Implications:** Open RAIL++-M permits commercial use (including live paid shows) with no revenue threshold, subject to standard responsible-AI use restrictions (no illegal/harmful use, propagate use restrictions if redistributing weights).
- **Legal Caveat (important for diligence):** Neither repo includes a standalone `LICENSE.md` text file (only the HF metadata tag), and `sdxs-512-0.9` lists `stabilityai/sd-turbo` as its teacher DM while `sdxs-512-dreamshaper` lists `Lykon/dreamshaper-8` / `dreamshaper-8-lcm` (SD 1.5 lineage, CreativeML OpenRAIL-M). From a clean-room licensing perspective, **`IDKiro/sdxs-512-dreamshaper` avoids any SD-Turbo teacher lineage ambiguity**. ([HF `sdxs-512-0.9`](https://huggingface.co/IDKiro/sdxs-512-0.9), [HF `sdxs-512-dreamshaper`](https://huggingface.co/IDKiro/sdxs-512-dreamshaper))

#### Diffusers Loading Code & One-Step Settings
From the official `IDKiro/sdxs-512-dreamshaper` and `IDKiro/sdxs-512-0.9` model cards ([HF `sdxs-512-dreamshaper`](https://huggingface.co/IDKiro/sdxs-512-dreamshaper), [HF `sdxs-512-0.9`](https://huggingface.co/IDKiro/sdxs-512-0.9)):

```python
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL

repo = "IDKiro/sdxs-512-dreamshaper"
pipe = StableDiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16)
# Optional: swap in the full VAE stored in subfolder "vae_large" if TAESD softens details too much:
# pipe.vae = AutoencoderKL.from_pretrained(repo, subfolder="vae_large", torch_dtype=torch.float16)
pipe.to("cuda")

# Required one-step settings: 1 step, CFG disabled (guidance_scale=0)
image = pipe(
    prompt="a golden meditating Buddha statue glowing with fractal light",
    num_inference_steps=1,
    guidance_scale=0,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
```

- **Important `sdxs-512-0.9` fp16 caveat:** The `sdxs-512-0.9` card explicitly warns that its bundled TAESD *"may produce low-quality images when `weight_type` is `float16`"* and defaults its example to `torch.float32` (or loading `vae_large`). `sdxs-512-dreamshaper` defaults to `torch.float16`. ([HF `sdxs-512-0.9`](https://huggingface.co/IDKiro/sdxs-512-0.9), [HF `sdxs-512-dreamshaper`](https://huggingface.co/IDKiro/sdxs-512-dreamshaper))

#### VRAM & Speed on 4 GB / GTX 1650-Class Cards
- **Parameter & Weight Footprint:** Total safetensors params are **315.7M** (`sdxs-512-dreamshaper`) / **328.2M** (`sdxs-512-0.9`) — roughly **~630–660 MB** of fp16 weights on GPU, plus activations + PyTorch CUDA context (~1.5–2.5 GB peak VRAM in fp16; *exact peak VRAM on GTX 1650 Ti Max-Q with PyTorch 2.x is unverified*). ([HF API `sdxs-512-dreamshaper`](https://huggingface.co/api/models/IDKiro/sdxs-512-dreamshaper), [HF API `sdxs-512-0.9`](https://huggingface.co/api/models/IDKiro/sdxs-512-0.9))
- **Paper Latency (Datacenter GPU):** The SDXS paper (`arXiv:2403.16627`) reports **~100 FPS** (~9–10 ms total: ~1.1 ms text encoder, ~6.3 ms pruned 319M UNet, ~1.7 ms tiny decoder) on a single high-end GPU (~30× faster than 50-step SD 1.5). ([arXiv:2403.16627](https://arxiv.org/abs/2403.16627))
- **GTX 1650 / 4 GB-Class Speed:** No peer-reviewed or official benchmark exists specifically for the 35W **GTX 1650 Ti Max-Q** (*exact ms/frame unverified*). Community reports for GTX 1650 4 GB place 1-step SDXS-512 generation at roughly **0.5–1.5 seconds per 512×512 image** (roughly ~0.7–2 FPS, *unverified*; faster than 1-step SD-Turbo because the UNet is 319M vs 866M parameters, roughly ~2–2.5× lighter compute). ([arXiv:2403.16627](https://arxiv.org/abs/2403.16627), [GitHub `IDKiro/sdxs`](https://github.com/IDKiro/sdxs))

#### Evidence of Recognizable Subject Quality (Portraits, Statues, Animals) vs. Abstract Textures
- **Quantitative Benchmark (MS-COCO 2017 5K, Paper Table 3):** ([arXiv:2403.16627](https://arxiv.org/abs/2403.16627))
  - **SDXS-512 (1 step, 319M UNet):** FID **28.18**, CLIP Score **32.81**
  - **SD-Turbo (1 step, 865M UNet):** FID **26.50**, CLIP Score **33.07**
  - **SD v1.5 (50 steps, 860M UNet):** FID **24.3**, CLIP Score **31.8**
  - **Tiny SD (16 steps, 324M UNet):** FID **31.16**, CLIP Score **31.06**
- **Qualitative & Community Evidence:**
  - `sdxs-512-dreamshaper` explicitly *"sacrifices diversity for better image generation quality"* (trained without focusing on FID, distilled from DreamShaper 8). Its card showcases a crisp portrait (*"a close-up picture of an old man standing in the rain"*), and `sdxs-512-0.9` showcases a female portrait (*"portrait photo of a girl..."*). ([HF `sdxs-512-dreamshaper`](https://huggingface.co/IDKiro/sdxs-512-dreamshaper), [HF `sdxs-512-0.9`](https://huggingface.co/IDKiro/sdxs-512-0.9))
  - **Strengths for VJing:** Strong on single, centered, stylized subjects (portraits, glowing icons, stylized animals) and psychedelic/painterly textures; `sdxs-512-dreamshaper-sketch` adds real-time ControlNet conditioning so you can lock a recognizable silhouette (e.g., a Buddha statue or portrait mask) while morphing textures. ([GitHub `IDKiro/sdxs`](https://github.com/IDKiro/sdxs))
  - **Weaknesses:** Strict 1-step trajectory straightening causes **mode/layout collapse** (narrower pose variety, tendency toward centered front-facing compositions), smoothed micro-textures (animal fur, feathers, skin pores look slightly painterly/plastic), and frequent distortion on **hands** or complex multi-subject prompts. ([HF `sdxs-512-dreamshaper`](https://huggingface.co/IDKiro/sdxs-512-dreamshaper), [arXiv:2403.16627](https://arxiv.org/abs/2403.16627))
  - **Specific "Buddha statue" subject benchmark:** No published evaluation specifically tests Buddha statues on SDXS-512 (**unverified** — test directly with the team's exact prompts and compare `sdxs-512-dreamshaper` vs. `Hyper-SD15` 2–4 step).

---

### 1.3 SD-Turbo (`stabilityai/sd-turbo`) — License & Suitability for Commercial Live Shows

- **Exact License:** **`Stability AI Community License Agreement` (Last Updated: July 5, 2024)**, present directly in the repo's `LICENSE.md`. ([SD-Turbo LICENSE.md](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md))
- **Commercial Revenue Threshold:** **Less than USD $1,000,000 per year** in aggregate annual gross revenue (covering Licensee and Affiliates, *"regardless of whether that revenue is generated directly or indirectly from the Stability AI Materials or Derivative Works"*). Below $1M/year, commercial use — including paid live VJ shows — is **royalty-free**. At or above $1M/year, the Community License terminates and you must obtain a paid Enterprise License (`https://stability.ai/enterprise`). ([SD-Turbo LICENSE.md](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md))
- **Mandatory Commercial Obligations (Section 3 & 4a):**
  1. **Registration required:** *"If You are using or distributing the Stability AI Materials for a Commercial Purpose, You must register with Stability AI at `https://stability.ai/community-license`."* ([SD-Turbo LICENSE.md](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md))
  2. **Attribution if distributing a product/service:** Provide a copy of the Agreement, include `"This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved"` in a `Notice` file, and prominently display `"Powered by Stability AI"` on related website/UI/docs. ([SD-Turbo LICENSE.md](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md))
  3. **Outputs & AUP:** As between you and Stability AI, **you own the generated outputs** (Section 4c(iii)). Must follow the [Stability AI Acceptable Use Policy](https://stability.ai/use-policy) and may not use outputs to train competing foundational models. ([SD-Turbo LICENSE.md](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md))
- **Technical Specs & Subject Quality:**
  - Distilled from **Stable Diffusion 2.1** via Adversarial Diffusion Distillation (ADD); **865.9M UNet parameters** (~1.25B total pipeline with OpenCLIP-H + VAE), 1–4 steps, `guidance_scale=0.0`, native `512×512`. ([HF `sd-turbo`](https://huggingface.co/stabilityai/sd-turbo), [HF API `sd-turbo`](https://huggingface.co/api/models/stabilityai/sd-turbo), [arXiv:2311.17042](https://arxiv.org/abs/2311.17042))
  - For `img2img`, `int(num_inference_steps * strength)` must be `>= 1` (e.g., `num_inference_steps=2, strength=0.5` runs 1 step). ([HF `sd-turbo`](https://huggingface.co/stabilityai/sd-turbo))
  - **Subject quality:** In the ADD human-preference study, 1-step SD-Turbo beat LCM-LoRA SD 1.5 and LCM-LoRA SDXL on image quality and prompt alignment (`arXiv:2311.17042`), and scores higher CLIP/FID than SDXS-512 on COCO-5K (33.07 / 26.50). **However**, the SD-Turbo model card explicitly warns: *"Faces and people in general may not be generated properly"* — inheriting SD 2.1's well-known weakness on human portraits compared to community SD 1.5 fine-tunes. ([HF `sd-turbo`](https://huggingface.co/stabilityai/sd-turbo), [arXiv:2311.17042](https://arxiv.org/abs/2311.17042))

---

### 1.4 Comparison Table: Small One/Few-Step Models (< ~1B Denoiser) for 4 GB Turing

| Model / HF Repo | Denoiser / Total Params | Steps (`guidance_scale`) | Est. VRAM (fp16, 512px) | License (Commercial?) | Subject Quality Evidence & 4 GB Turing Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`IDKiro/sdxs-512-dreamshaper`** ([HF](https://huggingface.co/IDKiro/sdxs-512-dreamshaper)) | ~315.7M total (pruned UNet + CLIP + TAESD) | **1** (`0.0`) | **~1.5–2.5 GB** *(unverified exact TU117)* | `openrail++` (**Yes**, no rev cap) | **Top speed + headroom pick.** Distilled from DreamShaper 8; good stylized portraits/subjects, narrow diversity, softer micro-detail. Leaves plenty of VRAM for OpenGL. |
| **`IDKiro/sdxs-512-0.9`** ([HF](https://huggingface.co/IDKiro/sdxs-512-0.9)) | ~328.2M total (319M UNet) | **1** (`0.0`) | ~2.0–3.0 GB *(fp32 rec. for TAESD)* | `openrail++` (**Yes**, SD-Turbo teacher) | Superseded by `sdxs-512-dreamshaper`. COCO FID 28.18 / CLIP 32.81 ([arXiv:2403.16627](https://arxiv.org/abs/2403.16627)). Author warns fp16 TAESD degrades quality. |
| **`stabilityai/sd-turbo`** ([HF](https://huggingface.co/stabilityai/sd-turbo)) | 865.9M UNet / ~1.25B total | **1–4** (`0.0`) | ~2.5–3.5 GB *(~2.2–2.8 GB w/ TAESD)* | Stability AI Community (**Yes <$1M/yr**, reg. req.) | Strong general prompt adherence (COCO FID 26.50 / CLIP 33.07; beats LCM-1.5 in human eval, [arXiv:2311.17042](https://arxiv.org/abs/2311.17042)). Weak on human faces (SD 2.1 base). Tight in 4 GB with OpenGL. |
| **Hyper-SD on SD 1.5** (`ByteDance/Hyper-SD` `Hyper-SD15-{1,2,4}steps-lora` + SD 1.5 / DreamShaper 8) ([HF](https://huggingface.co/ByteDance/Hyper-SD)) | 860M UNet / ~1.06B total *(0 overhead after `fuse_lora()`)* | **1** (`TCDScheduler`, `0.0`) or **2–4** (`DDIM` trailing, `0.0`) | ~2.5–3.2 GB *(~2.1–2.7 GB w/ TAESD)* | **CreativeML Open RAIL++-M** for SD1.5/SDXL ([LICENSE](https://huggingface.co/ByteDance/Hyper-SD/blob/main/LICENSE.md)) (**Yes**, no rev cap) | **Best recognizable subject quality in 4 GB.** SOTA 1–8 step distillation ([arXiv:2404.13686](https://arxiv.org/abs/2404.13686)); fuses into any SD 1.5 portrait/realism fine-tune + ControlNet compatible. |
| **LCM-LoRA on SD 1.5** (`latent-consistency/lcm-lora-sdv1-5` + SD 1.5 / DreamShaper 7) ([HF](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)) | 860M UNet (+67.5M LoRA, fused) | **2–8** (sweet spot **4**, `LCMScheduler`, CFG `0–2.0`) | ~2.5–3.2 GB *(~2.1–2.7 GB w/ TAESD)* | `openrail++` (**Yes**) | Proven img2img & ControlNet ecosystem ([arXiv:2311.05556](https://arxiv.org/abs/2311.05556)), but noticeably softer/blurrier than Hyper-SD15 at 1–2 steps. |
| **BK-SDM-Small / Tiny** (`nota-ai/bk-sdm-small`) ([HF](https://huggingface.co/nota-ai/bk-sdm-small)) | 490M UNet (660M total) / 330M UNet (500M total) | **25–50** *(not few-step distilled)* | ~1.5–2.2 GB | `creativeml-openrail-m` (**Yes**) | Pruned SD 1.4; COCO-30K FID 16.98 / CLIP 0.2677 at 25 steps ([arXiv:2305.15798](https://arxiv.org/abs/2305.15798)). Too many steps for live use without custom LCM/Hyper distillation (*unverified*). |
| **PixArt-Sigma 512 / PixArt-LCM** (`PixArt-alpha/PixArt-Sigma-XL-2-512-MS`, `PixArt-LCM-XL-2-1024-MS`) ([HF Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-512-MS), [HF LCM](https://huggingface.co/PixArt-alpha/PixArt-LCM-XL-2-1024-MS)) | 610.9M DiT + **4.7B T5-XXL encoder** | 14–20 (Sigma) / **4** (LCM, CFG `0.0`) | **>10 GB native** *(needs CPU offload / 4-bit T5 for 4 GB)* | `openrail++` *(card notes research Direct Use)* | **Poor fit for live 4 GB.** T5-XXL text encoder blows past 4 GB VRAM; CPU offloading per prompt adds multi-second stalls. |
| **SANA-0.6B / SANA-Sprint 0.6B** (`Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`) ([HF](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers)) | 0.6B Linear DiT + **2.6B Gemma-2-2B-IT** + DC-AE (~7.7 GB repo) | **1–4** (1 step SOTA) | **~8–9 GB native** *(requires offload/quant for 4 GB)* | Apache 2.0 + [Gemma Terms](https://ai.google.dev/gemma/terms) | **Avoid on 4 GB Turing.** Incredible quality (7.59 FID / 0.74 GenEval at 1 step, [arXiv:2503.09641](https://arxiv.org/abs/2503.09641)), **but BF16-native** (fp16 overflows to NaN/black on Turing without fp32 submodules) AND Gemma-2-2B alone exceeds 4 GB in fp16! |
| **DMD2** (`tianweiy/DMD2`) ([HF](https://huggingface.co/tianweiy/DMD2)) | SDXL UNet (~2.56B) | 1 or 4 (`0.0`) | >6 GB fp16 | **CC-BY-NC-SA-4.0 (NO commercial)** | Disqualified on both **4 GB VRAM** and **non-commercial license**. ([arXiv:2405.14867](https://arxiv.org/abs/2405.14867)) |

---

### 1.5 Speedups for SD-Turbo (and SD 1.5 / SDXS) on Turing (GTX 1650 Ti Max-Q, 4 GB, Windows)

| Optimization | How It Works | Realistic Gain on GTX 1650 Ti Max-Q (TU117, 4 GB, Windows) | Caveats / VRAM Impact |
| :--- | :--- | :--- | :--- |
| **`torch.float16` + PyTorch 2.x SDPA** (`AttnProcessor2_0`) ([Diffusers Docs](https://huggingface.co/docs/diffusers/main/en/optimization/fp16), [PyTorch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)) | Half-precision weights/compute (2:1 FP16 rate on Turing) + fused memory-efficient attention. | **Baseline mandatory (~2× faster & ~50% less VRAM vs fp32).** | Default in PyTorch 2.x; no extra Windows build tools required. Watch for rare fp16 NaN in SD 2.1 attention (rare with SDPA). |
| **TAESD (`madebyollin/taesd`)** ([HF TAESD](https://huggingface.co/madebyollin/taesd), [Diffusers AutoencoderTiny](https://huggingface.co/docs/diffusers/main/en/api/models/autoencoder_tiny)) | Replaces ~83M-param standard VAE decoder with a tiny ~2.4M-param distilled autoencoder (MIT license). | **Massive win on 4 GB:** Cuts VAE decode latency from ~50–150 ms to **<10 ms** and avoids ~500–800 MB VAE activation spikes (*exact TU117 ms unverified*). | Slight softness on micro-textures/fine edges — usually invisible or desirable under psychedelic shaders and projector optics. |
| **`channels_last` (`torch.channels_last`)** ([Diffusers FP16 Docs](https://huggingface.co/docs/diffusers/main/en/optimization/fp16), [Diffusers Memory Docs](https://huggingface.co/docs/diffusers/main/en/optimization/memory)) | Converts UNet & VAE tensors to NHWC layout to avoid cuDNN layout transpose overhead. | **Modest ~5–15% gain** (*exact TU117 benchmark unverified*; Diffusers docs explicitly note some models/GPUs can run slower — must benchmark). | Zero extra VRAM; trivial 2-line test (`pipe.unet.to(memory_format=torch.channels_last)`). |
| **`torch.compile` (Inductor + `triton-windows`)** ([Diffusers FP16 Docs](https://huggingface.co/docs/diffusers/main/en/optimization/fp16), [`triton-windows`](https://github.com/woct0rdho/triton-windows)) | Fuses elementwise/norm kernels and optionally captures CUDA Graphs (`mode="reduce-overhead"`). | **~10–25% realistic gain** on TU117 (Diffusers reports 20–30% on higher-end GPUs; *exact TU117 Windows gain unverified*). | **Windows friction:** Requires Visual Studio 2022 C++ Build Tools (`cl.exe`) + community `triton-windows` wheel. 30–90s cold JIT compile. **`mode="reduce-overhead"` (CUDA Graphs) reserves extra VRAM**, risking OOM on 4 GB alongside OpenGL! Test `mode="default"` or `compile_repeated_blocks()` first. |
| **TensorRT (static FP16 engine)** ([StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)) | Ahead-of-time graph fusion + cuDNN/CUDA kernel autotuning for SM 7.5. | **~20–40% gain** over eager PyTorch SDPA (*exact TU117 benchmark unverified*; much smaller than RTX 1.5–2.5× gains because TU117 has **no Tensor Cores**). | High maintenance on Windows: strict CUDA/TRT version coupling, 5–20 min engine build per fixed resolution, static shapes, and engine workspace can squeeze 4 GB VRAM. |
| **StreamDiffusion** ([GitHub](https://github.com/cumulo-autumn/StreamDiffusion), [arXiv:2312.12491](https://arxiv.org/abs/2312.12491)) | Combines Stream Batch + Residual CFG + Stochastic Similarity Filter + IO Queues + TAESD + optional TensorRT (106.2 FPS txt2img / 93.9 FPS img2img on RTX 4090). | **Realistic GTX 1650 Ti Max-Q throughput: ~1–4 FPS** at 512×512 1-step SD-Turbo (*unverified* on 35W Max-Q). | **Major 4 GB trap:** Stream Batch multiplies UNet batch size by `len(t_index_list)` → instant OOM on 4 GB for >1 step! For 1-step (`t_index_list=[0]`, `cfg_type="none"`), batch size is 1 anyway — better to implement prompt-embedding caching + TAESD + async worker thread directly in modern `diffusers` rather than pulling StreamDiffusion's legacy PyTorch 2.1 stack. |

---

## 2. Nano Banana via the `agy` (Antigravity) CLI

### 2.1 Official Gemini Image Model Names ("Nano Banana" Family — Nov 2026)

From Google's official Gemini API documentation ([Image Generation Docs](https://ai.google.dev/gemini-api/docs/image-generation), [Models Directory](https://ai.google.dev/gemini-api/docs/models)):

| Marketing Name | Exact Gemini API Model ID | Status (Nov 2026) | Key Capabilities & Resolutions | Paid API Price (1K / 2K / 4K) |
| :--- | :--- | :--- | :--- | :--- |
| **Nano Banana 2.1** | `gemini-nano-banana-2.1` | **New / Recommended Workhorse** (Oct 2026) | Improved realism & typography, **up to 14 reference images** (4 chars / 10 objects), configurable Thinking (`minimal`/`medium`/`high`), Google Web/Image Search grounding; `1K`/`2K`/`4K`. ([Model Page](https://ai.google.dev/gemini-api/docs/models/gemini-nano-banana-2.1)) | **$0.0336** / $0.0504 / $0.113 *(50% off in Batch)* |
| **Nano Banana 2 Lite** | `gemini-3.1-flash-lite-image` | **Stable — Fastest / Cheapest** (Jun 2026) | **Targets sub-2 second end-to-end latency**, interleaved text+image editing, `1K` only (`1024px`, 14 aspect ratios); not optimized for multi-reference or long multi-turn chains. ([Model Page](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite-image)) | **$0.0336** (1K only; cheapest input/thinking tokens: $0.25/1M) |
| **Nano Banana 2** | `gemini-3.1-flash-image` | **Stable** (Feb 2026; used inside Antigravity) | `0.5K`/`1K`/`2K`/`4K`, multi-ref consistency, Google Search grounding. ([Model Page](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-image)) | $0.045 (0.5K) / $0.067 (1K) / $0.101 (2K) / $0.151 (4K) |
| **Nano Banana Pro** | `gemini-3-pro-image` | **Stable — Premium Reasoning** (Nov 2025) | Heavy Thinking core for complex layouts, brand consistency, studio typography; `1K`/`2K`/`4K`. ([Model Page](https://ai.google.dev/gemini-api/docs/models/gemini-3-pro-image)) | $0.134 (1K/2K) / $0.24 (4K) |
| **Nano Banana** | `gemini-2.5-flash-image` | **Legacy** | Original Nano Banana; Google recommends migrating to `gemini-3.1-flash-lite-image`. ([Image Gen Docs](https://ai.google.dev/gemini-api/docs/image-generation)) | Legacy *(migrate to 2 Lite)* |

- *(Note: Preview endpoints such as `-preview` image IDs have been superseded by the stable IDs above; `imagen-4.0-generate` is shut down. All Nano Banana images carry an always-on invisible **SynthID** watermark, plus C2PA metadata on Nano Banana 2 Lite.)* ([Models Directory](https://ai.google.dev/gemini-api/docs/models), [SynthID](https://deepmind.google/technologies/synthid/))

---

### 2.2 Non-Interactive Image Generation with the `agy` CLI (Print / Headless Mode)

#### Architecture Reality Check
`agy` is an **agentic CLI**, not a bare image-API wrapper ([Antigravity Headless Docs](https://antigravity.google/docs/cli/headless)):
- The `--model <slug>` flag selects the **agent reasoning model** (listed by `agy models`, e.g., `gemini-4-hydrogen-low`, `gemini-4-neon-high`), **NOT** the image model. Passing an image model slug like `gemini-nano-banana-2.1` to `agy --model` exits `1` with `ERROR` (`invalid model selection`). ([Headless Docs](https://antigravity.google/docs/cli/headless))
- Per [Antigravity Models](https://antigravity.google/docs/models) ("Additional models"), Antigravity's built-in generative image tool is **not customizable** and uses **Nano Banana 2** under the hood (which the agent invokes via its image generation tool / built-in `image-generator` subagent, saving PNGs to disk). ([Antigravity Models](https://antigravity.google/docs/models), [Nano Banana Pro Blog](https://antigravity.google/blog/nano-banana-pro))

#### Exact Non-Interactive Flags & Command
Key headless flags documented in [Headless Mode](https://antigravity.google/docs/cli/headless) and [CLI Reference](https://antigravity.google/docs/cli/reference):
- `-p` / `--print` / `--prompt "<text>"`: Run a single prompt non-interactively, print response to `stdout` (diagnostics to `stderr`), and exit.
- `--output-format json` (or `stream-json` for live NDJSON tool/progress events, or `text`).
- `--model gemini-4-hydrogen-low --effort low`: Fastest, lowest-quota reasoning model to just dispatch image generation and copy the file.
- `--mode=accept-edits` (auto-approves workspace file writes) or `--dangerously-skip-permissions` (auto-approves all tool calls, including copying outside the current directory; otherwise headless mode soft-denies unapproved actions).
- `--add-dir <dir>`: Add target output directory to workspace scope (needed if saving outside `cwd` without `allowNonWorkspaceAccess: true`).
- `--print-timeout 120s`: Prevents hanging indefinitely on network stalls (`0s` default = infinite wait).
- `--disable-slash-commands`: Prevents prompts starting with `/` from triggering slash commands.

**One-shot command (text-to-image → PNG at given path):**
```powershell
agy -p "Generate a 16:9 psychedelic golden Buddha statue framed by sacred geometry, and save the resulting PNG to exactly C:\path\to\EntheoControl\frames\hero_01.png. Reply only with the saved file path." `
  --model gemini-4-hydrogen-low `
  --effort low `
  --add-dir "C:\path\to\EntheoControl\frames" `
  --dangerously-skip-permissions `
  --output-format json `
  --print-timeout 120s
```
*(Pre-authenticate once interactively via `agy`; headless mode reuses cached credentials and exits with `authentication required` if unauthenticated.)* ([Headless Docs](https://antigravity.google/docs/cli/headless))

**Persistent multi-turn process (eliminates per-call CLI startup overhead):**
Launch `agy --input-format stream-json --output-format stream-json --model gemini-4-hydrogen-low --effort low --dangerously-skip-permissions` and write NDJSON `{"event":"user","message":{"content":"..."}}` lines to `stdin`, reading until each `{"event":"result",...}` line on `stdout`. ([Headless Docs](https://antigravity.google/docs/cli/headless))

#### Image Input (`img2img` / Edit from a Reference Frame)
- **Does it work?** **Yes**, via **file paths in the text prompt** — NOT via binary/base64 blocks in `stream-json`.
- **Critical `stream-json` limitation:** Official docs explicitly state that in `--input-format stream-json`, **`text` is the only supported content block type**; submitting any image/non-text block immediately terminates the session with exit code `1`. ([Headless Docs](https://antigravity.google/docs/cli/headless))
- **Working non-interactive pattern:** Place your reference frame PNG on disk (in cwd or an `--add-dir` folder) and reference its exact path in the prompt:
  `"Using C:\...\ref_frame.png as the reference image, transform it into a bioluminescent fractal temple while preserving the silhouette, and save the edited PNG to C:\...\out_frame.png."`
  The agent reads/passes the reference file to its Nano Banana image-editing tool. *(Interactive TUI also supports `Ctrl+V` pasting of PNG/JPEG/GIF/WebP/BMP/TIFF/SVG, but that is interactive-only.)* ([Prompting Docs](https://antigravity.google/docs/cli/prompting))

#### Typical Latency, Output Format, and Quotas
- **Output format:** **PNG** (`image/png`). Base64-decoded PNG from the Interactions API (`interaction.output_image.data`) or written directly as `.png` on disk by Antigravity's image tool. Always embeds an invisible **SynthID** watermark. ([Gemini Image Gen Docs](https://ai.google.dev/gemini-api/docs/image-generation))
- **Typical latency:**
  - **Direct Gemini API:** `gemini-3.1-flash-lite-image` (Nano Banana 2 Lite) explicitly targets **sub-2 second end-to-end latency** at 1K. ([Nano Banana 2 Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite-image)) Nano Banana 2 / 2.1 typically run several seconds (~3–10 s depending on resolution, thinking level, and server load; *exact p50/p95 latency for 2.1 unverified*).
  - **Via `agy -p`:** Adds CLI process startup + agent reasoning turn + subagent/tool orchestration + file save overhead (expect roughly **~8–25 seconds** per cold `agy -p` call; *exact measured `agy -p` image latency unverified*). Persistent `stream-json` mode cuts the cold startup penalty.
- **Individual Account Quotas & Limits:**
  - **Antigravity (`agy`) personal accounts** ([Antigravity Plans](https://antigravity.google/docs/plans), [`/usage` Docs](https://antigravity.google/docs/cli/commands/usage)):
    - **Free & Google AI Plus:** Meaningful quota, refreshed **weekly**.
    - **Paid Google AI Pro:** Generous quota refreshed every **5 hours** up to a **weekly cap**, plus optional paid AI Credit overages (`useG1Credits: true` in `settings.json`). Standard Pro *trials* get lower quotas.
    - **Google AI Ultra:** Highest 5-hour and weekly quotas.
    - **No fixed numeric image limit is published** — quotas scale dynamically with agent work and system capacity (inspect live remaining quota interactively via `/usage` or `/quota`). BYOK is **not** supported to expand Antigravity quotas. ([Plans](https://antigravity.google/docs/plans))
  - **Direct Gemini API (with `GEMINI_API_KEY`):** All four current Nano Banana models (`2.1`, `2`, `2 Lite`, `Pro`) list **Free Tier: Not available** on their official pricing pages and require a linked **Paid Cloud Billing account** (Tier 1 spend limit **$10 per rolling 10 min**, $250/mo cap; Tier 2 **$200 per 10 min**). Image models are rate-limited by **RPM, TPM, RPD, and IPM (Images Per Minute)** visible per-project in Google AI Studio. ([Rate Limits](https://ai.google.dev/gemini-api/docs/rate-limits), [Nano Banana 2.1](https://ai.google.dev/gemini-api/docs/models/gemini-nano-banana-2.1))

---

### 2.3 Commercial Use Terms & Critical Warning on Wrapping `agy`

- **Commercial use of Nano Banana outputs (Gemini API):** Under the [Gemini API Additional Terms of Service](https://ai.google.dev/gemini-api/terms) (Effective March 23, 2026):
  - Google explicitly states the API is *"for developers building with Google AI models for professional or business purposes"* and **"Google won't claim ownership over that content"** (*Use of Generated Content*). Commercial use in live shows is permitted, subject to Google's [Prohibited Use Policy](https://policies.google.com/terms/generative-ai/use-policy) and applicable law (note: raw AI outputs may lack copyright protection depending on jurisdiction).
  - Under **Paid Services** (active Cloud Billing project), Google **does not** use your prompts, reference images, or outputs to improve Google products. ([Gemini API Terms](https://ai.google.dev/gemini-api/terms))
- **CRITICAL ToS WARNING for Antigravity (`agy`) Automation:** The [Google Antigravity Additional Terms of Service](https://antigravity.google/terms) state:
  > *"You must not abuse, harm, interfere with, or disrupt the Service. This includes, but is not limited to, using the Service in connection with products not provided by us. **Using third party software, tools, or services to access the Service (e.g. using OpenClaw with Antigravity OAuth) is a breach of this Agreement. Such actions may be grounds for suspension or termination of your Antigravity and/or Gemini CLI accounts.**"*
  - Additionally, personal Antigravity accounts allow Google to use Interactions for product/ML improvement unless opted out in settings. ([Antigravity Terms](https://antigravity.google/terms))
- **Recommended Alternative for the Live VJ Engine:** Call Nano Banana directly via the official **`google-genai` Python SDK** (`client.interactions.create(model="gemini-3.1-flash-lite-image", ...)`) or Google's official **`gemini-api` CLI** (`gemini-api image "prompt" -i ref.png --out out.png`) with a Paid `GEMINI_API_KEY`. This is **5–10× lower latency** (sub-2s on 2 Lite), gives exact model/aspect/resolution control, avoids agent failure modes mid-show, costs ~$0.034/frame (~$3.36 for 100 hero frames), and is 100% compliant for commercial production! ([Gemini Image Gen Docs](https://ai.google.dev/gemini-api/docs/image-generation))

---

## 3. Photosensitive Safety Limits for Live Visuals

### 3.1 Authoritative Standards Compared (ITU-R BT.1702, Ofcom, Harding FPA, WCAG 2.3.1)

The global broadcast and digital safety standards for preventing Photosensitive Epilepsy (PSE) seizures converge on the same core physiological thresholds ([ITU-R BT.1702](https://www.itu.int/rec/R-REC-BT.1702), [WCAG 2.2 SC 2.3.1 Understanding](https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html), [Harding FPA](https://www.hardingfpa.com/)):

| Parameter | ITU-R BT.1702 / Ofcom / Harding FPA (Broadcast) | WCAG 2.1 / 2.2 SC 2.3.1 (Digital / Web) |
| :--- | :--- | :--- |
| **Flash Frequency Limit** | **≤ 3 flashes in any 1-second period** (a flash = 2 opposing transitions: dark→bright→dark or bright→dark→bright). Successive flashes separated by ≥ 9 frames (at 50 Hz) / ≥ 333 ms are safe. ([ITU-R BT.1702](https://www.itu.int/rec/R-REC-BT.1702)) | **≤ 3 general flashes and/or ≤ 3 red flashes in any 1-second period.** ([WCAG 2.2 SC 2.3.1](https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html)) |
| **General Flash Luminance Change** | **ΔL ≥ 20 cd/m²** when darker state **< 160 cd/m²**; when darker state **≥ 160 cd/m²** (HDR / high-brightness), **Michelson contrast `(L_high − L_low) / (L_high + L_low) ≥ 1/17` (5.88%)**. ([ITU-R BT.1702](https://www.itu.int/rec/R-REC-BT.1702), [WCAG 2.2 SC 2.3.1](https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html)) | Opposing change in linearized sRGB relative luminance **ΔL ≥ 0.10 (10% of max 1.0)** where darker state **`L_dark < 0.80`** (`L = 0.2126R + 0.7152G + 0.0722B`). ([WCAG 2.2 SC 2.3.1](https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html)) |
| **Screen Area Threshold** | Combined concurrent flashing area **> 25%** of displayed screen area triggers violation. ([ITU-R BT.1702](https://www.itu.int/rec/R-REC-BT.1702)) | Combined concurrent flashing area **> 0.006 steradians** within any **10° visual field** (**> 25% of any 10° visual field**; 341 × 256 CSS px on 1024×768 reference). ([WCAG 2.2 SC 2.3.1](https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html)) |
| **Saturated Red Flash Rule** | **Zero-tolerance / special rule:** Any transition to or from a **saturated red** is treated as potentially harmful *irrespective of overall luminance delta*, capped at ≤ 3 flashes/s over > 25% area. ([ITU-R BT.1702](https://www.itu.int/rec/R-REC-BT.1702), [Harding FPA](https://www.hardingfpa.com/)) | **ISO 9241-391 working definition (in WCAG 2.1/2.2):** Pair of opposing transitions where either state has **`R / (R + G + B) ≥ 0.8`** (linear RGB) **AND** state difference **`> 0.2` (unitless)** in the **CIE 1976 UCS (`u', v'`) chromaticity diagram** (older WCAG 2.0 proxy: `Δ(max(0, R−G−B) × 320) > 20`). ([WCAG 2.2 SC 2.3.1](https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html)) |
| **Spatial Regular Patterns** | High-contrast stripes/checkerboards/bars with **> 5 light-dark pairs** (cycles) occupying > 25% of screen area — especially if flashing, reversing direction, or oscillating. ([Harding FPA](https://www.hardingfpa.com/), [ITU-R BT.1702](https://www.itu.int/rec/R-REC-BT.1702)) | Exempts only fine balanced patterns (e.g., white noise or checkerboards with squares **< 0.1°** visual angle on a side). ([WCAG 2.2 SC 2.3.1](https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html)) |

*(Official UK broadcast guidance: [Ofcom Guidance Note for Licensees on Flashing Images and Regular Patterns (`gn_flash.pdf`)](https://www.ofcom.org.uk/siteassets/resources/documents/tv-radio-and-on-demand/broadcast-guidance/gn_flash.pdf); note automated HTTP fetch returned 403 from Ofcom's WAF, so exact clause-level numbers above are cross-verified directly against [ITU-R BT.1702](https://www.itu.int/rec/R-REC-BT.1702) and [WCAG 2.2 SC 2.3.1](https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html).)*

---

### 3.2 Concrete Limits for an Automated VJ Assistant Controlling a Projector Show

**Why projectors are strictly riskier than monitors:** A dark room + high-lumen projector filling a wall/stage subtends **30°–70°+** of an audience member's field of view (far exceeding the 10° reference window), and dark-adapted eyes are more sensitive to luminance jumps. Therefore, **assume every full-frame or large-zone effect automatically exceeds the 25% / 0.006 sr area limit** and enforce hard global temporal and chromatic caps in the VJ engine itself (never rely solely on LLM prompt instructions):

1. **Hard Strobe / Pulse Frequency Cap:**
   - **Maximum strobe rate: `2.5 Hz`** (conservative safety margin below the 3.0 Hz legal/clinical ceiling; never allow > 3.0 Hz).
   - **Beat-sync trap:** A 120–180 BPM track has 2–3 beats/sec — **quarter-note strobing at > 150 BPM or eighth-note strobing at > 75 BPM violates 3 Hz!** Clamp audio-reactive/BPM strobe triggers to a **minimum inter-flash interval of `400 ms`** (≤ 2.5 Hz) or **≤ 3 opposing luminance reversals in any rolling `1000 ms` buffer**.
2. **Blackout / Whiteout / Cut Guard:**
   - Enforce a **minimum hold duration ≥ `350 ms`** (recommended `500 ms`) or a smooth **fade ramp ≥ `250 ms`** on blackout/whiteout/scene-cut triggers so rapid automated toggling (`black → bright → black → bright`) can never generate > 3 flashes/second.
   - Apply a **per-frame global luminance monitor** on a downsampled (e.g., 64×36) copy of the final composited output framebuffer computing linearized sRGB `L = 0.2126*R + 0.7152*G + 0.0722*B`: if a rolling 1.0 s window already contains **2 opposing reversals with `|ΔL| ≥ 0.08`** (conservative below WCAG's 0.10), engage an automatic **low-pass temporal damper** (`alpha` blend with previous frame) until the window clears!
3. **Saturated Red Protection (Mandatory for Psychedelic Visuals):**
   - Compute `red_ratio = R_lin / (R_lin + G_lin + B_lin + 1e-6)` on the downsampled output (or in the color-grading shader).
   - Whenever an effect/palette enters **`red_ratio ≥ 0.75`** (safety margin below WCAG's `0.80`):
     - **Hard-disable strobe and rapid hue cycling** (cap modulation frequency to **≤ 1.0 Hz** and luminance/chroma slew rate so full-red-to-black transitions take **≥ 500 ms**), OR
     - Automatically **desaturate / mix in ≥ 20–25% green/blue** (shifting pure crimson slightly toward warm amber/magenta) during beat-reactive pulses so `R / (R+G+B)` stays safely below `0.75`.
4. **High-Contrast Pattern & Tunnel Shader Guard:**
   - For kaleidoscope, radial tunnel, checkerboard, and concentric-ring shaders: cap contrast (`L_max - L_min < 0.5`) or apply soft anti-aliasing/blur whenever spatial frequency creates **> 5 high-contrast light-dark bands** across the projection surface, and **prohibit alternating phase-inversion** (checkerboard polarity flipping) at > 2 Hz.
5. **Kill-Switch & Audience Notice:**
   - Keep a hardware/hotkey **smooth-dim panic button** (fades output to 10% warm low-contrast ambient over 300 ms — *not* a harsh cut) and post standard venue photosensitive-visuals signage if any flashing effects above 1 Hz are used.

---

## 4. Laptop Power & Heat During Live GPU Rendering

### 4.1 ThinkPad Optimus + i9-10885H Thermal Bottleneck

Reviews of the **ThinkPad X1 Extreme Gen 3 / P1 Gen 3** (Intel Core i9-10885H 8C/16T, 2.4 GHz base / 5.3 GHz Turbo, 14nm Comet Lake-H + 35W GTX 1650 Ti Max-Q) document that Lenovo allows short CPU bursts up to ~90 W in a thin ~1.7 kg chassis with a **shared dual-fan heatpipe cooling assembly**. Under combined CPU + GPU load, the CPU spikes to 95–100 °C and both chips throttle severely (CPU dropping to ~35 W sustained) while saturating the shared heatsink. ([Notebookcheck ThinkPad X1 Extreme Gen 3 Review](https://www.notebookcheck.net/ThinkPad-X1-Extreme-Gen3-2020-Lenovo-s-Premium-Multimedia-Laptop-with-GTX-1650-Ti-Max-Q-in-Review.513512.0.html))

### 4.2 Windows Power Plan & Disabling Intel Turbo Boost ("99% Trick" vs. `PERFBOOSTMODE`)

- **Effect of Maximum Processor State = 99% (`PROCTHROTTLEMAX`, GUID `bc5038f7-23e0-4960-96da-33abaf5935ec`):** ([Microsoft Q&A — Maximum Processor State](https://learn.microsoft.com/en-us/answers/questions/3771576/how-to-enable-maximum-processor-state))
  - Dropping Windows *Processor power management → Maximum processor state* (Plugged in) from `100%` to **`99%`** prevents Windows from requesting Intel Turbo Boost P-states, capping the i9-10885H at its **2.4 GHz base clock**.
  - **Thermal effect:** Typically lowers CPU package temperatures by **~15–25 °C** (*exact °C delta varies by chassis/ambient, unverified on your specific unit*), drastically reducing heat dumped into the shared heatpipes so the 35W GTX 1650 Ti Max-Q maintains steady GPU boost clocks without thermal throttling.
  - **Tradeoff for Python/diffusers:** OpenGL rendering and GPU-bound diffusion steps are barely affected at 2.4 GHz, while one-time CPU tasks (model weight loading from disk, Python startup, CPU-offloaded encoders) run slower.
- **Cleaner Windows Alternative (`PERFBOOSTMODE`, GUID `be337238-0d82-4146-a960-4f3749d470c7`):** ([Microsoft Q&A — Processor Performance Power Settings](https://learn.microsoft.com/en-us/answers/questions/3966055/set-processor-performance-energy-preference-policy))
  - Unhide *Processor performance boost mode* in an Administrator terminal:
    `powercfg -attributes SUB_PROCESSOR be337238-0d82-4146-a960-4f3749d470c7 -ATTRIB_HIDE`
  - Keep Maximum Processor State at `100%`, and set **Processor performance boost mode** (Plugged in) to **`Disabled`** (for coolest sustained show) or **`Efficient Enabled`** (allows brief low-voltage boost for Python spikes without aggressive 90W thermal spikes).
- **Power Plan & AC Adapter:** Always run on the **full-wattage Lenovo AC adapter (135W+)** and set Windows Power Mode to **Balanced** (not "Best Performance", which keeps CPU clocks pinned unnecessarily high during idle periods).

### 4.3 NVIDIA Control Panel / NVIDIA App Settings for a Live Projector Rig

| Setting | Recommended Live-Show Value | Why / Thermal & Rendering Impact |
| :--- | :--- | :--- |
| **Max Frame Rate** *(NVIDIA Control Panel → Manage 3D Settings → Program Settings → `python.exe`)* | **60 FPS** (match projector refresh) or **30 FPS** (if running concurrent heavy diffusion) | Hard-caps GPU rendering so the GTX 1650 Ti Max-Q doesn't burn 100% utilization / 35W rendering wasted frames beyond projector refresh. Directly cuts GPU heat and leaves CUDA headroom for `diffusers`. ([NVIDIA 441.87 Release Notes](http://us.download.nvidia.com/Windows/441.87/441.87-win10-win8-win7-release-notes.pdf)) |
| **Low Latency Mode** *(NVIDIA Control Panel)* | **`Off`** or **`On`** (avoid `Ultra`) | `On` caps CPU pre-rendered frames to 1; `Ultra` submits just-in-time. For a VJ projector engine sharing the GPU with PyTorch CUDA inference, `Off` (or `On`) gives smoother frame pacing than `Ultra`, which can induce micro-stutter when CUDA kernels momentarily preempt graphics (*exact OpenGL+CUDA interaction unverified — benchmark on rig*). ([NVIDIA Gamescom 2019 Driver](https://www.nvidia.com/en-us/geforce/news/gamescom-2019-game-ready-driver/), [NVIDIA System Latency Guide](https://www.nvidia.com/en-us/geforce/guides/system-latency-optimization-guide/)) |
| **WhisperMode** *(NVIDIA App / GeForce Experience)* | **OFF** | WhisperMode dynamically caps FPS and GPU power to hit an acoustic fan target on AC ([NVIDIA WhisperMode](https://www.nvidia.com/en-eu/geforce/technologies/whisper-mode/)). **Disable for live shows** so NVIDIA doesn't unexpectedly throttle frame rate mid-performance; use explicit **Max Frame Rate** instead. |
| **BatteryBoost** *(NVIDIA App / GeForce Experience)* | **N/A on AC** (do not run live shows on battery) | Only activates when **unplugged**, aggressively capping FPS/power to extend battery runtime ([NVIDIA BatteryBoost](https://www.nvidia.com/en-eu/geforce/technologies/battery-boost/)). Never run a live GPU VJ + diffusion rig on battery. |
| **Power Management Mode** *(Control Panel)* | **`Normal`** (test `Prefer maximum performance` only if clock-down stutter appears) | `Normal` allows clocks to relax between frames when FPS-capped, lowering average heat. |
| **Optimus Display Routing** *(Control Panel → Configure Surround, PhysX)* | Plug projector into a **dGPU-wired port** (HDMI or USB-C/TB3 DP wired directly to GTX 1650 Ti) | On ThinkPad X1 Extreme / P1 Gen 3, external HDMI/DisplayPort outputs are typically wired directly to the NVIDIA dGPU (*verify in NVIDIA PhysX topology diagram*), bypassing Intel iGPU Optimus frame-copy overhead and avoiding cross-GPU sync stutter. ([Notebookcheck Review](https://www.notebookcheck.net/ThinkPad-X1-Extreme-Gen3-2020-Lenovo-s-Premium-Multimedia-Laptop-with-GTX-1650-Ti-Max-Q-in-Review.513512.0.html)) |

### 4.4 NVIDIA OpenGL VSync Busy-Spin (100% CPU Core) & Fixes

- **Does NVIDIA's Windows OpenGL driver busy-spin a CPU core during VSync?**
  - **Yes — this is a well-documented driver behavior.** Inside `nvoglv64.dll`, NVIDIA's Windows OpenGL driver frequently implements `wglSwapBuffers` VSync waits (and `glFinish()`) using an active **spin-wait / hybrid polling loop** rather than immediately sleeping the thread in the Windows kernel, in order to minimize vblank wake-up jitter ([NVIDIA Developer Forums — OpenGL VSync SwapBuffers 100% CPU](https://forums.developer.nvidia.com/t/opengl-vsync-swapbuffers-100-cpu-core/29086), [StackOverflow — 100% CPU on OpenGL VSync](https://stackoverflow.com/questions/21925313/100-cpu-utilization-when-using-vsync-opengl)). On a laptop like the i9-10885H, pinning a single core to 100% triggers sustained single-core Turbo heat and fan noise even when the OpenGL shader itself takes < 2 ms!
  - **Role of "Threaded optimization":** NVIDIA's *Threaded optimization* (`Auto`/`On`) moves OpenGL driver command submission onto an internal worker thread. With custom Python/OpenGL loops or shared contexts, this frequently triggers thread contention and makes `SwapBuffers` spin aggressively at 100% CPU (or causes micro-stutters). ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/opengl-vsync-swapbuffers-100-cpu-core/29086))
- **Concrete Fixes (apply in order and verify in Task Manager / Process Explorer):**
  1. **Disable Threaded Optimization per-app:** In **NVIDIA Control Panel → Manage 3D Settings → Program Settings → `python.exe`**, set **Threaded optimization = `Off`**. *(Note: behavior varies across NVIDIA driver branches — verify CPU usage of `python.exe` before and after; use NVIDIA Profile Inspector if Control Panel hides or ignores the profile.)*
  2. **Explicit CPU-side Frame Pacing before `SwapBuffers`:** Keep VSync enabled (`swap_interval = 1`), measure frame render time, and sleep cleanly in the OS until ~1–2 ms before the next vblank (or, on Windows DWM borderless fullscreen, synchronize with Win32 **`DwmFlush()`** which blocks cleanly via the OS compositor without busy-spinning!). That reduces `nvoglv64.dll`'s spin window from ~14 ms down to ~1 ms. ([Microsoft Win32 `DwmFlush` Docs](https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/nf-dwmapi-dwmflush))
  3. **Single-Thread OpenGL Discipline & No `glFinish()`:** Create the OpenGL context, render, and call `SwapBuffers` on **one dedicated thread** (pass newly generated diffusion frames from a background worker via lock-free queue / PBO), and **never call `glFinish()`** per frame (use `glFenceSync` / `glClientWaitSync` if synchronization is needed).
  4. **Windows vs. Linux Env-Var Note:** The NVIDIA environment variables `__GL_YIELD=USLEEP` and `__GL_THREADED_OPTIMIZATIONS=0` are **Linux X11/GLX driver controls** and do **NOT** take effect on Windows `nvoglv64.dll`. ([NVIDIA Linux OpenGL Env Variables](https://us.download.nvidia.com/XFree86/Linux-x86_64/550.54.14/README/openglenvvariables.html))
