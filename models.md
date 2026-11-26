# Models

EntheoControl ships no model weights. Each optional feature downloads (or asks you to download) its model the first
time it is used. Licenses and attribution: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Crowd camera (models/camera/, download by hand)

From `https://storage.googleapis.com/mediapipe-models/`:

| File (save into `models/camera/`) | Path under the URL above | SHA-256 of the tested file |
|---|---|---|
| `selfie_segmenter_landscape.tflite` | `image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite` | `490e9ea734313e0de10fa0cd9e3c6133e36ea4db2b7a49bde9ef019f72796b8e` |
| `pose_landmarker_lite.task` | `pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task` | `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` |

The `latest` paths can change upstream. If a hash differs, the file is a newer release: it should still work.

```powershell
New-Item -ItemType Directory -Force models\camera
$u = "https://storage.googleapis.com/mediapipe-models"
Invoke-WebRequest "$u/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite" -OutFile models\camera\selfie_segmenter_landscape.tflite
Invoke-WebRequest "$u/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile models\camera\pose_landmarker_lite.task
Get-FileHash models\camera\*
```

## Local AI (downloaded automatically into the Hugging Face cache on first use)

| Used by | Model | Source |
|---|---|---|
| `ai_worker.py`, `ai_gen.py` (default) | SD-Turbo | https://huggingface.co/stabilityai/sd-turbo |
| `--base sd15-lcm` | Stable Diffusion v1.5 + LCM-LoRA | https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5, https://huggingface.co/latent-consistency/lcm-lora-sdv1-5 |
| fast VAE for both | TAESD | https://huggingface.co/madebyollin/taesd |
| art depth maps | Depth Anything V2 Small | https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf |
| Dreamify | torchvision GoogLeNet, `GoogLeNet_Weights.IMAGENET1K_V1` | downloaded by torchvision into the torch hub cache |
