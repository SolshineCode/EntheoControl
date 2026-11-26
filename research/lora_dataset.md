# Legal Dataset & Low-VRAM Training Plan for a Visionary Beadwork / DMT LoRA

> **Critical Architecture Warning for `ai_gen.py`:**
> `stabilityai/sd-turbo` is distilled from **Stable Diffusion 2.1** (OpenCLIP ViT-H/14, 1024-dim cross-attention), **NOT** SD 1.5 (CLIP ViT-L/14, 768-dim).
> - An **SD 1.5 LoRA will NOT load** into `stabilityai/sd-turbo` — PyTorch will raise a tensor shape mismatch error.
> - **Pick one consistent base before training:**
>   - **Path A (Recommended for ecosystem & 4 GB training):** Train an **SD 1.5** LoRA (`runwayml/stable-diffusion-v1-5` or `stable-diffusion-v1-5/stable-diffusion-v1-5`), and run `ai_gen.py` with **SD 1.5 + `latent-consistency/lcm-lora-sdv1-5`** (2–4 steps, TAESD, circular padding).
>   - **Path B (Keep `sd-turbo` 1-step):** Train your LoRA directly against **SD 2.1 base** (`stabilityai/stable-diffusion-2-1-base`) or `stabilityai/sd-turbo`.

---

## 1. Building a 40–100 Image Training Set Legally & Ethically

### Strict Boundary
- **Do NOT scrape or train on copyrighted works by living artists** (Alex Grey, Android Jones, Johnathan Singer, TAS Visuals, Chris Dyer, etc.). Use their art strictly as visual study for color harmony, symmetry, black-background contrast, and composition.

### Recommended Clean Sources (Aim for 60–80 Diverse 512×512 Images)

1. **100% Self-Generated Procedural Renders (Best Source — 30–50 images):**
   - **Blender 4.5 LTS Geometry Nodes** (already being installed via `winget`): build 5–8 reusable procedural node graphs (concentric phyllotaxis sphere/bead rings, toroidal sacred-geometry lattices, crystalline mandalas, hyperbolic tessellations) rendered on **pure `#000000` black backgrounds** with emissive jewel-tone materials (turquoise, magenta, gold, indigo, emerald). Vary seed, symmetry fold (6/8/12/16), camera zoom, and palette to export 30–50 crisp 512×512 or 1024×1024 stills.
   - **GLSL / `live_visuals.py` & Fractal Flame tools:** Capture high-contrast stills from your own GLSL shaders or open-source fractal flame renderers (e.g., **Apophysis 7x** / **JWildfire** — 100% original renders you own outright).
   - *Why this wins:* Zero copyright ambiguity, seamless dark backgrounds tailor-made for projection, and exact beadwork density.

2. **Public Domain (CC0) Museum Archives (15–25 images):**
   - **The Met Museum Open Access (CC0 1.0):** https://www.metmuseum.org/about-the-met/policies-and-documents/open-access — Always toggle the **"Open Access"** filter before downloading! Rich sources of historical geometric beadwork, Islamic *girih* tilework & star polygons, Persian carpets, stained glass rose windows, and pre-Columbian textiles.
   - **Smithsonian Open Access (CC0):** https://www.si.edu/openaccess — Filter explicitly for **CC0** items (textiles, ornamental metalwork, natural-history symmetry).
   - **Ernst Haeckel — *Kunstformen der Natur* (1904, Public Domain):** High-res plates on Wikimedia Commons (https://commons.wikimedia.org/wiki/Kunstformen_der_Natur) — radiolarians, diatoms, sea anemones, and jellyfish with extraordinary biological radial symmetry.
   - **Cultural respect note on Indigenous motifs:** Historical Wixárika (Huichol *nierika*) and Shipibo-Conibo (*kené*) textiles in museum collections may be legally public domain/CC0, *but* living communities regard these patterns as sacred cosmology. Check each item's Open Access badge (skip restricted items), avoid presenting AI outputs as authentic indigenous craft, and prefer abstracting geometric principles into your own procedural Blender/shader beadwork.

3. **Open-Licensed Generative Fractal Archives (10–20 images):**
   - **Electric Sheep** (Scott Draves): https://electricsheep.org/ — Community-evolved fractal flame frames/packs distributed under Creative Commons (verify individual sheep/pack license — **CC-BY** vs **CC-BY-NC** — on the Electric Sheep server or Internet Archive before including if you plan commercial gigs).

---

## 2. Dataset Prep & Captioning Tips (for Seamless Projection Textures)

- **Resolution:** Crop/render square **512 × 512** PNGs (no letterboxing, no watermarks, no text).
- **True-black discipline:** Keep dark/black backgrounds in at least 70% of your training images so the LoRA naturally learns projection-ready glowing geometry instead of muddy gray fills.
- **Captioning (`.txt` file matching each `.png`):**
  - Pick a unique rare trigger token (e.g., `visbead`).
  - Keep captions concise and descriptive so you can steer color/shape at inference time:
    - Example: `visbead, concentric glowing beadwork mandala, turquoise gold magenta jewel tones, pure black background, symmetrical sacred geometry, intricate micro-beads`
    - Example: `visbead, luminous neon geometric filigree lattice, deep indigo and emerald, dark void background, fractal tessellation`
  - **Repeat count:** With 60 images, use `10` repeats × `10` epochs (~600–1,500 total steps at batch size 4 effective) — abstract/psychedelic textures converge fast and overfit if pushed past ~2,000 steps.

---

## 3. Training on 4 GB VRAM Locally vs. Free Cloud GPU

### Option A: Local Training on GTX 1650 Ti 4 GB (`kohya-ss/sd-scripts` or `bmaltais/kohya_ss`)
Repos:
- CLI scripts: https://github.com/kohya-ss/sd-scripts
- GUI wrapper: https://github.com/bmaltais/kohya_ss

Minimum-VRAM SD 1.5 LoRA settings that fit under 4 GB VRAM:
```bash
accelerate launch --num_cpu_threads_per_process=2 train_network.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --train_data_dir="lora_data/img" \
  --output_dir="loras" \
  --output_name="visbead_sd15_r16" \
  --resolution="512,512" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --mixed_precision="fp16" \
  --save_precision="fp16" \
  --optimizer_type="AdamW8bit" \
  --network_module="networks.lora" \
  --network_dim=16 \
  --network_alpha=8 \
  --cache_latents \
  --cache_latents_to_disk \
  --gradient_checkpointing \
  --sdpa \
  --max_train_epochs=10 \
  --save_every_n_epochs=2 \
  --lr_scheduler="cosine" \
  --learning_rate=1e-4 \
  --text_encoder_lr=5e-5
```

**Important 4 GB GTX 1650 Ti Caveats:**
1. **Close all GPU apps** (Chrome/Edge, TouchDesigner, Resolume, Discord) before training. If VRAM exceeds ~3.7 GB, Windows silently spills into **Shared GPU Memory (system RAM)** and slows training by 10×–20× without throwing an OOM error.
2. **GTX 16xx FP16 NaN risk during backward pass:** Some Turing GTX 16-series cards produce `loss=nan` under `--mixed_precision="fp16"` during LoRA training, and full `fp32` training will **not** fit in 4 GB VRAM. If you hit `loss=nan` locally, immediately move training to Kaggle/Colab below.

### Option B: Free Cloud GPU Fallback (Strongly Recommended!)
Training is a one-time offline job (15–25 minutes) and outputs a tiny **18–36 MB `.safetensors`** file that you drop into the project's `loras\` folder and run locally forever.
- **Kaggle Notebooks (Best free tier):** Offers **30 hours/week free GPU** (NVIDIA **T4 × 2** or **P100**, 16 GB VRAM each — native Tensor Cores on T4, zero FP16 NaN or 4 GB OOM headaches). Upload your zipped 60-image dataset as a private Kaggle Dataset and run `kohya-ss/sd-scripts` or a Kaggle-adapted Kohya notebook.
- **Google Colab (`hollowstrawberry/kohya-colab`):** https://github.com/hollowstrawberry/kohya-colab — Well-maintained one-click Kohya LoRA trainer notebook for SD 1.5 (and SDXL). Note that Google Colab's free tier enforces stricter GPU session caps than Kaggle, so Kaggle is usually more reliable for uninterrupted runs.

---

## Sources
- Kohya `sd-scripts`: https://github.com/kohya-ss/sd-scripts
- Kohya_ss GUI (`bmaltais/kohya_ss`): https://github.com/bmaltais/kohya_ss
- Hollowstrawberry Kohya Colab Notebooks: https://github.com/hollowstrawberry/kohya-colab
- The Met Museum Open Access (CC0 1.0): https://www.metmuseum.org/about-the-met/policies-and-documents/open-access
- Smithsonian Open Access (CC0): https://www.si.edu/openaccess
- Ernst Haeckel *Kunstformen der Natur* (Wikimedia Commons, Public Domain): https://commons.wikimedia.org/wiki/Kunstformen_der_Natur
- Electric Sheep Fractal Archive: https://electricsheep.org/
- Latent Consistency Model LoRA for SD 1.5 (`latent-consistency/lcm-lora-sdv1-5`): https://huggingface.co/latent-consistency/lcm-lora-sdv1-5
