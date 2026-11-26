# Third-party notices

EntheoControl itself is licensed under the GNU Affero General Public License v3.0 ([LICENSE](LICENSE)). This file lists
the third-party code adapted into the source, the Python packages it uses, and the AI models it downloads at runtime,
each with its license. Licenses were checked against the upstream pages linked below (2026-11-26).

## 1. Code adapted into this repository

### 1.1 Inigo Quilez: 2D distance functions, smooth minimum, Voronoi, sine hash (MIT)

Where: `beings.py` (`smin`, `sdSeg`, `sdCone`, `sdEll`, `sdBox`, `sdVes`, `sdPoly8`, `sdTri`, `h11`, `h21`; full
notice also at the top of that file), `fx_stack.py` (`hash2` and the animated cell search in `voronoiGlass`),
`kene_real.py` (`hash21`), `stack.py` (`h11`), and the one-line `random` mod-shape preview in `panel.html`.
The box distance used inside `math_layers.py` is the same standard formula.

Sources: https://iquilezles.org/articles/distfunctions2d/, https://iquilezles.org/articles/smin/,
https://iquilezles.org/articles/voronoilines/ and the Shadertoy demos linked from them (for example
https://www.shadertoy.com/view/ldl3W8). The article pages do not state a license themselves. The linked Shadertoy demos
carry "The MIT License, Copyright © 20xx Inigo Quilez" headers, as reproduced in for example Blender's
`source/blender/blenlib/intern/noise.cc` (Voronoi, "Original code is under the MIT License, Copyright (c) 2013 Inigo
Quilez") and Material Maker's SDF nodes. We treat all of it as MIT:

```
The MIT License
Copyright (c) 2013-2024 Inigo Quilez

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

### 1.2 David Hoskins: "Hash without Sine" (MIT)

Where: `filters.py` (`hash12`, `hash22`), `live_visuals.py` (`hash21`), `fx_stack.py` (`hash12`), `art_layer.py`
(`hash21`). Source: https://www.shadertoy.com/view/4djSRW (the license header below is the one distributed with the
code, as reproduced verbatim in for example taisei-project/taisei `shader/lib/hash.glslh`).

```
Hash without Sine
MIT License...
Copyright (c)2014 David Hoskins.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 1.3 Code that was replaced (no third-party code remains)

Shadertoy's default license is CC BY-NC-SA 3.0, which is not compatible with the AGPL. These snippets had no license
that allowed AGPL distribution, so they were rewritten (2026-11-26):

- `fx_stack.py` `hexGrid`: previously followed the widely shared "two offset rectangular grids, pick the nearer centre"
  hexagon snippet. It is now written from the lattice geometry (axial coordinates, cube-coordinate rounding, three edge
  normals). It returns the same cell ids and edge distances (checked numerically on 200,000 points: identical ids, edge
  distance within 2e-9), so the hex domes look the same.
- `art_layer.py` `hash21`: previously the `fract(p*vec2(123.34, 456.21))` hash. It is now Hoskins' MIT hash (1.2). The
  noise it drives has the same statistics. Individual noise values differ.
- `fx_stack.py`: two inline `fract(sin(dot(p, vec2(12.9898, 78.233)))*43758.5453)` random calls now use Hoskins'
  `hash12` (1.2). The same share of cells is lit, but a different set of them.

### 1.3b Scene transitions (`transitions.py`): original

The transition shaders (doors, sliding doors, curtains, iris, portal, pattern reveal, shatter, peel, dissolve, wipe,
zoom through) were written for this project from the geometry (hinged-panel perspective inverse, gauge functions for the
iris shapes, a fold reflection for the peel). No code from gl-transitions (MIT) or Shadertoy was ported. The per-cell
random value in SHATTER is the generic `fract(sin(x) * k)` construction with this project's own constants. If a
gl-transitions effect is ported later, its MIT notice goes here and in the shader's header.

### 1.4 TouchVDMX Explorer prototype (not included)

The four raymarched "3D decks" worlds, the blend modes and the particle swarm in `fx_stack.py` were reimplemented after
studying `touchvdmx_explorer.html`. That file is a single-page browser prototype that the project owner generated in
their own Google Gemini session and handed to the developers as a reference. It is not part of this repository and is
not distributed. The code in `fx_stack.py` is a new implementation in this engine (different renderer, structure,
camera, wrapping and colour handling), not a copy. Where it uses well-known building blocks, those are covered by
1.1 to 1.3 above.

### 1.5 Fonts, images, kené study photos

No fonts or images are bundled. `fonts/`, `textures/` and `art/` hold your own files. The kené study photos the
generators were designed against (`research/kene_reference/images/`, public domain / CC0 / CC BY / CC BY-SA museum and
archive photos) are not distributed. Their sources, credits and licenses are listed in section 5.1 of
`research/kene_reference/README.md`.

## 2. AI models downloaded at runtime (optional features)

No model weights are distributed with EntheoControl. They are downloaded on first use under their own licenses, which
you accept by downloading them. Download details: [models.md](models.md).

| Model | Used for | License | Source |
|---|---|---|---|
| SD-Turbo (`stabilityai/sd-turbo`) | default prompt-to-texture (`ai_worker.py`, `ai_gen.py`) | **Stability AI Community License** (see 2.1) | https://huggingface.co/stabilityai/sd-turbo, https://stability.ai/license |
| Stable Diffusion v1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`) | `--base sd15-lcm` only | CreativeML OpenRAIL-M (use restrictions in its Attachment A) | https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 |
| LCM-LoRA SD1.5 (`latent-consistency/lcm-lora-sdv1-5`) | `--base sd15-lcm` only | CreativeML OpenRAIL++-M | https://huggingface.co/latent-consistency/lcm-lora-sdv1-5 |
| TAESD (`madebyollin/taesd`) | fast VAE decoder | MIT | https://huggingface.co/madebyollin/taesd |
| Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`) | depth maps for uploaded art | Apache-2.0 (the Small model only: Base/Large are CC BY-NC 4.0) | https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf |
| GoogLeNet / Inception v1, `GoogLeNet_Weights.IMAGENET1K_V1` (torchvision) | Dreamify | torchvision code BSD-3-Clause. torchvision says pre-trained weights "may have their own licenses or terms and conditions derived from the dataset used for training" (ImageNet). Check that it fits your use | https://github.com/pytorch/vision#pre-trained-model-license |
| MediaPipe Selfie Segmenter (landscape) | crowd camera silhouettes | Apache-2.0 (Google model card) | https://huggingface.co/litert-community/MediaPipe-ImageSegmenter-Selfie-Landscape, https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter |
| MediaPipe Pose Landmarker (lite) | crowd camera pose | Apache-2.0 (Google model card) | https://huggingface.co/litert-community/MediaPipe-PoseLandmarker-Lite, https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker |
| Online plugin (off by default): Google Gemini image models via your own API key or the `agy` CLI, Claude Code CLI | assistant + image generation | provider terms of service. Your own account and key | https://ai.google.dev/gemini-api/terms, https://www.anthropic.com/legal |

### 2.1 SD-Turbo: Stability AI Community License

SD-Turbo is not covered by the AGPL. Its license (https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md,
"Last Updated: July 5, 2024") in short (read the full text):

- Research and non-commercial use (for example personal / hobby shows) is free.
- **Commercial use, for example a paid show or a commercial installation, requires registering at
  https://stability.ai/community-license.** It is free only while you and your affiliates make less than USD 1,000,000
  in annual revenue. Above that you need an enterprise license (https://stability.ai/enterprise).
- If you distribute the model or a product or service that uses it, include a copy of the license and a "Notice" text
  file with: "This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd.
  All Rights Reserved", and prominently display "Powered by Stability AI".
- Use must follow Stability AI's Acceptable Use Policy. Outputs must not be used to train other foundation models.

EntheoControl does not distribute SD-Turbo, so the project carries the attribution as a courtesy:
**Powered by Stability AI** (when the local AI worker runs with its default SD-Turbo model). This Stability AI Model is
licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved.

If you do not want these terms, run the AI worker with `--base sd15-lcm` (OpenRAIL) or do not use the AI features.

## 3. Python packages (installed by you from PyPI, not bundled)

Versions are the tested ones. See [requirements.txt](requirements.txt).

| Package | Version tested | License | Needed for |
|---|---|---|---|
| moderngl (+ glcontext) | 5.13.0 | MIT | engine (required) |
| glfw | 2.11.0 | MIT (bundles the GLFW library, zlib/libpng) | engine (required) |
| numpy | 2.4.6 | BSD-3-Clause (+ bundled 0BSD, MIT, Zlib, CC0) | required |
| pillow | 13.0.0 | MIT-CMU | required |
| SoundCard | 0.4.6 | BSD-3-Clause | audio input |
| psutil | 7.2.2 | BSD-3-Clause | health monitor |
| mido / python-rtmidi | 1.3.3 / 1.5.8 | MIT / MIT (RtMidi: MIT-style) | MIDI |
| imageio-ffmpeg | 0.8.1 | BSD-2-Clause. The bundled ffmpeg binary is GPL/LGPL, see https://github.com/imageio/imageio-ffmpeg | recorder, video clips |
| aalink | 0.2.3 | GPL-3.0-or-later (compatible with AGPL-3.0 under GPLv3/AGPLv3 section 13) | Ableton Link (optional) |
| SpoutGL | 0.1.2 | BSD-3-Clause (Spout2: BSD-2-Clause) | Spout output (optional) |
| nvidia-ml-py | 12+ | BSD-3-Clause | GPU stats (optional) |
| qrcode | 8.2 | BSD | console QR code (optional) |
| torch / torchvision | 2.14.1 / 0.29.1 (cu126) | BSD-3-Clause style (PyTorch license) / BSD-3-Clause | local AI (optional) |
| diffusers / transformers / accelerate / peft / safetensors / huggingface_hub | 0.43.0 / 5.24.1 / 1.16.0 / 0.22.0 / 0.8.0 / 2.4.0 | Apache-2.0 | local AI (optional) |
| opencv-contrib-python | 5.1.0.95 | Apache-2.0 | crowd camera, AUTO MAP (optional) |
| mediapipe | 1.2.3 | Apache-2.0 | crowd camera (optional) |
| pywinpty | 3.1.2 | MIT | online plugin agy backend (optional) |
| google-genai | any | Apache-2.0 | online plugin Nano Banana API backend (optional) |
| pytest / playwright | 9.2.0 / 1.65.2 | MIT / Apache-2.0 | tests |

`panel.html` loads nothing from the network (no CDN, no web fonts).
