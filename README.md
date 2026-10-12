# mlx-heart-swift

**HEART** — Hybrid Efficient Attention with Rank-factorized bias Transformer — by
[Philip Hofmann (Phips)](https://huggingface.co/Phips/HEART), ported PyTorch → Swift/MLX for Apple silicon and wrapped
as an [MLXEngine](https://github.com/xocialize/mlx-engine-swift) `imageUpscale` package. A 16.7 M-parameter HAT-iLN
window-attention super-resolution network, **trained only on a CC0 corpus**, with four released checkpoints:

| variant | upstream checkpoint | scale | role |
|---|---|---|---|
| `.fidelity` (default) | `heart_4x_otf_v2` | 4× | OTF fidelity — the best damaged-input arm on the Forge bench (blur, noise, JPEG), both oracles agreeing |
| `.sharp` | `heart_4x_otf_gan` | 4× | OTF GAN — sharpest real-world texture |
| `.clean` | `heart_4x_pretrain` | 4× | clean sources |
| `.clean2x` | `heart_2x` | 2× | the only native ×2 checkpoint |

Weights: [`mlx-community/HEART-fp16`](https://huggingface.co/mlx-community/HEART-fp16) (shipping lane, 33.7 MB per variant)
and [`mlx-community/HEART-fp32`](https://huggingface.co/mlx-community/HEART-fp32) (parity lane) — per-tensor exact
conversions of the upstream files, materialized per variant by the engine on first use.

```
mlx-heart-swift
├── Sources/HEARTMLX     engine-agnostic core: HEART, i-LN, RIB window attention, reflect padding, the PlaybackTier
│                        over the shared tile driver (mlx-realesrgan-swift) — no MLXToolKit dependency
├── Sources/MLXHEART     HEARTUpscalePackage + HEARTConfiguration (the MLXEngine ModelPackage)
├── Sources/Smoke        heart-smoke — the CLI gate lane (S0–S7) and the real-engine smoke
├── Tests                core structure/op tests (XCTest, CPU stream) · wrapper conformance (C0–C14, MAT, CAN, INF)
└── oracle               make_shim.sh · convert_weights.py · dump_goldens.py · upload_weights.py · reports/
```

## Use

```swift
import MLXServeCore, MLXHEART

let engine = MLXServeEngine()
let id = try await engine.register(HEARTUpscalePackage.registration,
                                   configuration: HEARTConfiguration(variant: .fidelity, quant: .fp16))
let out = try await engine.run(ImageUpscaleRequest(image: image), package: id)      // 4× (2× via .clean2x)
```

`HEARTConfiguration` selects the variant, the precision lane (`.fp16` default, `.fp32` parity — `BudgetAware` drops fp32 to
fp16 under a tight budget), and the tile geometry (`wholeFrameMaxPixels`, `inputTileSize`, `tileOverlap`). A sub-native
`scale` is honored by downsampling the native result; `appliedScale` reports what ran.

⚠️ HEART's i-LN normalises over the **whole image**, so the tiled path is not bit-identical to the whole-frame path (each
tile is normalised by its own statistics). The default geometry is measured, not inherited — see `PORTING-SPEC.md` H5.

Cancellation: the entry checkpoint, one per tile, and one per residual group (six per forward), with `RunProgress` at
the same seams.

## 16 bits end to end (0.2.0)

The surface declares `ToolDescriptor.rawImageFormats = [.rawBGRA8, .rawRGBA16]` (MLXToolKit contract 1.52.0, AB-T-0229),
the mlx-realplksr-swift 0.2.0 boundary.
- A `.rawRGBA16` image comes back `.rawRGBA16`, and a PNG or TIFF deeper than 8 bits comes back as a 16-bit PNG.
- Both run through the shared tile driver's 16-bit path (mlx-realesrgan-swift ≥ 0.8.1), with the same tiling as 8 bits.
  The sub-native resample runs at 16 bits too.
- 8-bit inputs take the 8-bit path unchanged.

⚠️ **The fp16 lane needed more than a converter.** HEART has no float32 base under its output: `conv_last` writes
the image directly, in half precision. A 16-bit run on the shipping lane came back with only ~4,000 levels per
channel, and it failed the G-6 gate on 3 of 5 images (smooth-area flat-neighbour share 0.079 / 0.025 / 0.040 against
the interpolation's 0.052 / 0.015 / 0.023): the Float16 staircase AB-A-0158 predicted for a half boundary.
- So a 16-bit buffer runs `HEART.forward(_:keepingPrecision: true)`: the stem conv reads the float32 input, and the
  tail (`conv_after_body` → `conv_last`) runs in float32. The 36 residual blocks stay fp16.
- Distance from an all-float32 forward on the G-6 set: 0.11–0.14 of an 8-bit step mean on fp16, 0.02–0.05 with the
  float32 stem and tail.
- The tail's upsampler runs in 64 K-pixel row strips with a 2-row halo, which is exact by receptive field. Un-stripped,
  float32 at the output scale took a 512² tile's MLX peak from 3.6 to 7.1 GB. Stripped, it is 3.2 GB at the fp16
  forward's speed (1.8 s).

Gates (2026-10-11, M5 Max):
- **8-bit byte identity:** Release builds on the GPU, against v0.1.0, 4 images × {whole-frame, tiled} × {png ×4,
  png ×2, rawBGRA8 ×4}: 24/24 outputs byte-identical.
- **G-6 direct gate** (best, the AB-R-0489 set, through the Image Engine): the 16-bit output's smooth-area
  flat-neighbour share is 0.000–0.018, at or below the 16-bit interpolation's 0.015–0.165 on every image (the 8-bit
  model: 0.26–0.48). Its 56–65 k levels sit 0.20 of a step mean from the 8-bit output.

## Sub-native scale is a Lanczos reduction (0.2.1)

A `scale` below the native factor (×2 from the ×4 network) is now a vImage Lanczos reduction of the native frame,
at 8 or 16 bits (AB-T-0225).
- **The defect:** the CoreGraphics `.high` draw it replaces sampled on a corner-aligned grid. Content drifted
  ±0.26 px from the centre toward the edges.
- **What it cost:** on the G-4 drawn ladder (AB-R-0486), HEART `.fidelity`'s shipped ×2 lost 2.8 dB mean PSNR against a
  Lanczos reduction of the same ×4 frame, up to 6.3 dB on line art. Photos lost less.
- **Now:** the shipped ×2 matches PIL's Lanczos to 0.02 dB.
- **Test:** `resizeStaysRegisteredAcrossTheFrame` places blocks near both corners and the centre. It failed on 0.2.0
  (4.26 for 4.00, 247.74 for 248) and passes here at 8 and at 16 bits.
- **Unchanged:** ×4 and other native-scale outputs are untouched.

## Off mlx's channel-pad helper (0.2.2)

`conv_before_upsample` (180 → 64 channels, 3×3) is a `RoutedConv2d` now (AB-T-0214).
- **The defect:** mlx 0.32.2 serves a stride-1 conv2d with 5+ input channels that aren't a multiple of 16 through
  `pad_in_channels_conv_2D_gpu`. That helper frees its zero fill value while the GPU still reads it. Under a capped
  buffer cache (the Image Engine's 2 GiB) the process can die with `Invalid Resource`. Every HEART forward took it
  once, in the 8-bit tail and in the 16-bit float32 tail.
- **Now:** ExactConv pads the channels in the graph: 180 → 192, or 208 where 192 + 64 channels would land in mlx's
  lossy Winograd window, so the conv stays on implicit GEMM as before. `conv32` keeps a routed conv's route.
- **Same output:** a Release Image Engine build gave byte-identical ×4, ×2, 16-bit ×2 and drawn-content outputs
  against 0.2.1. The conv alone is bit-identical in fp16 and fp32.
- **Test:** `testConvBeforeUpsampleIsRoutedOffThePadHelper` (GPU against a CPU conv).

## Port

An isomorphic translation of `heart_arch.py` and the traiNNer-redux helpers it imports (same classes, same forward
order; NHWC; the `(3, heads, c)` QKV split, the non-wrapping reflect-shift, i-LN's inference semantics and the pre-scaled
SDPA are all encoded and probed). Parity against the upstream PyTorch code: every sub-op within 1e-5 relative,
end-to-end **127.5 dB** (`.fidelity`, fp32, CPU stream and Metal with `MLX_ENABLE_TF32=0`); the author's ONNX exports as
the second oracle. The shipping fp16 lane reads **67.6 dB** against that oracle and is SSIMULACRA2-indistinguishable
from fp32 on the Forge bench (Δ −0.01), at **416–450 ms per output megapixel** on M5 Max (head padded to 64 for the
fused attention kernel; ~4× RealPLKSR). The full gate table, the dtype and tiling studies and the lessons (MLX's naive
CPU reductions, torch's 1-ulp `sin`, fp16 reductions) are in [`PORTING-SPEC.md`](PORTING-SPEC.md).

Building the CLI gate lane: `xcrun swift build -c release --target HEARTSmoke` → `.build/out/Products/Release/heart-smoke`.

## Licence and credits

Apache-2.0 (port code and weights). HEART is by **Philip Hofmann (Phips)** — code and pretrained weights Apache-2.0,
trained only on [`Phips/lucid-cc0-v2-hc-512`](https://huggingface.co/datasets/Phips/lucid-cc0-v2-hc-512) (platform-declared
CC0: LUCID ← `nyuuzyou/pxhere` ← pxhere.com). The architecture builds on **HAT** (XPixelGroup, Apache-2.0, arXiv 2205.04437)
and **HAT-iLN** (arXiv 2504.06629), traiNNer-redux's `hat_iln_arch.py` (Apache-2.0), and **SST**'s RIB (arXiv 2603.06738).
See `NOTICE`.
