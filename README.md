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
