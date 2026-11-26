"""Local AI psychedelic texture generator for a 4 GB laptop GPU.

SD-Turbo (1-step) + TAESD tiny VAE + seamless (circular-padded) tiling, so the
output can be zoomed, scrolled and kaleidoscoped by live_visuals.py without seams.

Usage:
  python ai_gen.py --n 8                       # 8 textures with built-in prompts
  python ai_gen.py --prompt "..." --n 4 --seed 7
  python ai_gen.py --base sd15-lcm --n 8          # SD1.5 + LCM, 4 steps
  python ai_gen.py --base sd15-lcm --lora my_sd15_lora.safetensors   # SD1.5 LoRAs need sd15-lcm
  (SD-Turbo is SD 2.1 based: only SD2.1/Turbo LoRAs load on the default base)
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoPipelineForText2Image, AutoencoderTiny

PROMPTS = [
    "intricate huichol beadwork mandala, glowing jewel tones, turquoise magenta gold, black background, sacred geometry, ultra detailed",
    "shipibo kene pattern made of luminous neon lines, fractal, symmetrical, deep indigo and emerald, visionary art",
    "hyperbolic tessellation of glowing crystal jewels, dmt hyperspace, iridescent, intricate, black background",
    "visionary anatomy energy lattice, translucent human form, glowing chakras, translucent, fractal filigree, cosmic",
    "infinite fractal temple of light, interlocking sacred geometry, psychedelic, ornate, gold and violet glow",
    "swirling paisley fractal of luminous beads, ayahuasca vision, serpentine patterns, vivid jewel colors",
    "kaleidoscopic alien machine elves architecture, impossible geometry, glowing filigree, hyperdetailed",
    "bioluminescent mycelium mandala, neon fungal fractal network, deep black background, ultra detailed",
]


def make_seamless(module: torch.nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.padding_mode = "circular"


BASES = {
    # SD-Turbo is distilled from SD 2.1 -> needs SD2.1/SD-Turbo LoRAs. 1 step.
    "sd-turbo": dict(repo="stabilityai/sd-turbo", steps=1, cfg=0.0),
    # SD 1.5 + LCM-LoRA -> standard SD1.5 LoRAs (most community LoRAs) plug in. 4 steps.
    "sd15-lcm": dict(repo="stable-diffusion-v1-5/stable-diffusion-v1-5", steps=4, cfg=1.0),
}


def load_pipe(base: str, lora: str | None, dtype=torch.float16, vae_fp32: bool = False):
    cfg = BASES[base]
    pipe = AutoPipelineForText2Image.from_pretrained(
        cfg["repo"], torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None,
        safety_checker=None, requires_safety_checker=False,
    )
    pipe.vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesd", torch_dtype=torch.float32 if vae_fp32 else dtype)
    if base == "sd15-lcm":
        from diffusers import LCMScheduler
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")
        if lora:
            pipe.load_lora_weights(lora, adapter_name="style")
            pipe.set_adapters(["lcm", "style"], adapter_weights=[1.0, 0.8])
        pipe.fuse_lora()
    elif lora:
        pipe.load_lora_weights(lora)
    make_seamless(pipe.unet)
    make_seamless(pipe.vae)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    pipe = pipe.to("cuda")
    if vae_fp32 and dtype != torch.float32:
        # fp16 UNet latents -> fp32 tiny VAE (fixes GTX 16xx NaN at ~2.3 GB)
        orig = pipe.vae.decode
        pipe.vae.decode = lambda z, *a, **k: orig(z.float(), *a, **k)
    return pipe, cfg

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--base", default="sd-turbo", choices=list(BASES))
    ap.add_argument("--steps", type=int, default=None, help="default: 1 for sd-turbo, 4 for sd15-lcm")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--vae-fp32", action="store_true", help="first fix to try if images come out black (GTX 16xx)")
    ap.add_argument("--fp32", action="store_true", help="full fp32, last resort (tight on 4 GB)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "textures"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pipe, bcfg = load_pipe(args.base, args.lora, torch.float32 if args.fp32 else torch.float16, args.vae_fp32)
    steps = args.steps or bcfg["steps"]
    print(f"loaded in {time.time() - t0:.1f}s, VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    for i in range(args.n):
        prompt = args.prompt or PROMPTS[i % len(PROMPTS)]
        g = torch.Generator("cuda").manual_seed(seed + i)
        t = time.time()
        img = pipe(prompt, num_inference_steps=steps, guidance_scale=bcfg["cfg"],
                   height=args.size, width=args.size, generator=g).images[0]
        dt = time.time() - t
        arr = np.asarray(img)
        if arr.max() < 5:  # GTX 16xx fp16 NaN -> black image guard
            print(f"[{i}] WARNING: black/NaN image, try --vae-fp32, then --fp32")
        path = out / f"tex_{seed + i:010d}.png"
        tmp = path.with_suffix(".tmp.png")
        img.save(tmp)
        tmp.replace(path)  # atomic: live_visuals never sees a half-written file
        print(f"[{i}] {dt:.2f}s  {path.name}  :: {prompt[:60]}")
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
