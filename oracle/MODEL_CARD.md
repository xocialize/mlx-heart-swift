---
library_name: mlx
license: apache-2.0
license_link: https://huggingface.co/Phips/HEART/blob/main/LICENSE
pipeline_tag: image-to-image
base_model: Phips/HEART
datasets:
  - Phips/lucid-cc0-v2-hc-512
tags:
  - mlx
  - super-resolution
  - image-restoration
  - heart
  - hat
  - transformer
---

# {{REPO}}

**HEART** — Hybrid Efficient Attention with Rank-factorized bias Transformer, by Philip Hofmann (Phips)
([Phips/HEART](https://huggingface.co/Phips/HEART), Apache-2.0): a 16.7 M-parameter HAT-iLN window-attention
super-resolution network, trained **only** on the CC0 [`Phips/lucid-cc0-v2-hc-512`](https://huggingface.co/datasets/Phips/lucid-cc0-v2-hc-512)
corpus — converted to MLX in **{{LANE}}** for the Swift/MLX port
[`xocialize/mlx-heart-swift`](https://github.com/xocialize/mlx-heart-swift) (MLXEngine `imageUpscale`, the fidelity
stills tier). The four released checkpoints, per-tensor exact conversions of the upstream files at revision
`868878ce4c253a8061300f923b620fdc6edf090a` (`oracle/convert_weights.py`: upstream keys unchanged, conv weights
`(O,I,kH,kW)` → `(O,kH,kW,I)`, nothing else touched):

| file | upstream checkpoint | variant | scale | role |
|---|---|---|---|---|
| `heart_4x_otf_v2_{{LANE}}.safetensors` | `models/heart_4x_otf_v2.safetensors` | `.fidelity` (default) | 4× | OTF fidelity — the best damaged-input arm on the Forge bench |
| `heart_4x_otf_gan_{{LANE}}.safetensors` | `models/heart_4x_otf_gan.safetensors` | `.sharp` | 4× | OTF GAN — sharpest real-world output |
| `heart_4x_pretrain_{{LANE}}.safetensors` | `models/heart_4x_pretrain.safetensors` | `.clean` | 4× | official pretrain — clean sources |
| `heart_2x_{{LANE}}.safetensors` | `models/heart_2x.safetensors` | `.clean2x` | 2× | official 2× pretrain |
| `config.json` | — | — | — | architecture, variants, source sha256s, the fp16 dtype rule |

Lanes: `mlx-community/HEART-fp16` is the shipping lane (conv / linear tensors fp16; the i-LN and affine weight/bias and
the RIB implicit-net parameters stay fp32 — 220 of 748 tensors — and the port runs every reduction in fp32: the i-LN
statistics, the RIB position tables, the softmax, the global average pool and its squeeze/excite head);
`mlx-community/HEART-fp32` is the parity / reference lane. Each variant is a
separate file so a package pulls only the checkpoint it uses.

## Parity

Against the upstream PyTorch implementation (`heart_arch.py` + traiNNer-redux helpers, CPU fp32, same input), fp32 lane:
`.fidelity` **127.5 dB** PSNR at 128² (`.sharp` 125.3, `.clean` 122.5, `.clean2x` 124.0); every sub-op within 1e-5
relative. Against the author's ONNX exports under ONNX Runtime the port agrees exactly as far as torch itself does
(90.2 / 84.9 / 95.8 dB at 128²; the `heart_4x_pretrain` ONNX file does not reproduce its checkpoint, 7.8 dB, and ORT drifts
with image size). Details, the fp16 study and the tiling study: `PORTING-SPEC.md` in the port repo.

## Use with mlx-heart-swift

```swift
import MLXServeCore, MLXHEART
let engine = MLXServeEngine()
let id = try await engine.register(HEARTUpscalePackage.registration,
                                   configuration: HEARTConfiguration(variant: .fidelity, quant: .fp16))
let out = try await engine.run(ImageUpscaleRequest(image: image), package: id)   // 4× (2× via .clean2x)
```

The engine materializes the selected variant's file from this repo into its model store on first use (`WeightSourcing`).

## Licences and provenance

Apache-2.0 throughout: the HEART code and pretrained weights (Philip Hofmann / Phips), the HAT and HAT-iLN
architecture family (XPixelGroup, Apache-2.0; arXiv 2205.04437, 2504.06629), traiNNer-redux's `hat_iln_arch.py`
(Apache-2.0) and SST's RIB (arXiv 2603.06738). Training corpus: `Phips/lucid-cc0-v2-hc-512` — platform-declared CC0
(LUCID ← `nyuuzyou/pxhere` ← pxhere.com), which the re-host takes as governing. Credit the author when you use these
weights.
