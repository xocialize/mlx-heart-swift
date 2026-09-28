# mlx-heart-swift — porting spec (HEART → Swift/MLX)

Reference: `Phips/HEART` @ **868878ce4c253a8061300f923b620fdc6edf090a** (Apache-2.0, code + weights; trained only on
`Phips/lucid-cc0-v2-hc-512`, CC0 — provenance approval [[AB-D-0106]]). Plan: `mlxengine-forge/Docs/NERVE-HEART-PORT-PLAN.md`
§3 (phases H0–H7, traps §3.3, the tiling decision §3.4); evidence `NERVE-EVAL.md` §4.3 / §5.7 ([[AB-R-0370]]); task
[[AB-T-0189]]. Path B (novel architecture), PyTorch → Swift directly with per-sub-op goldens from the upstream code.

**THE oracle executes upstream code, never a reading of it:** `heart_arch.py` (Phips/HEART 868878ce) + the traiNNer-redux
helpers it imports (`hat_iln_arch.py`, the `iLN` class of `arch_util.py`, `registry.py` @ 686293190aca), rebuilt byte-for-byte
by `oracle/make_shim.sh` (the forge's script, copied), torch 2.13 CPU fp32. The **second oracle** is the author's own ONNX
exports under ONNX Runtime 1.30 (CPU).

## Sources and pins (sha256)

| file | sha256 |
|---|---|
| `models/heart_4x_otf_v2.safetensors` (`.fidelity`) | `1dd45db6cb28653e6374544613a6d895621d66afbf94972d30f9d918f9071c30` |
| `models/heart_4x_otf_gan.safetensors` (`.sharp`) | `f25403e371543693706088b497fe006ffba85de409c2db371ce104365b432157` |
| `models/heart_4x_pretrain.safetensors` (`.clean`) | `940c03a062fadeb4190726ddf509aa9a2e64de310549ea2147a5d9547b2809ab` |
| `models/heart_2x.safetensors` (`.clean2x`) | `7f225a1a80b9c1ddf260aafaf0968d217b3caa64a6568f190fa5586c38203fce` |
| `onnx/heart_4x_otf_v2_fp32_op17.onnx` | `2b0d73b363b5ec5fd0dbccb538d74c90c1ff344e4b1dde7c8fd0ec4ad5badb8f` |
| `onnx/heart_4x_otf_gan_fp32_op17.onnx` | `19e8700ce947bca61e5c6f1bbb7f12a3e8b2ca7ed13af7e5c801937df12084a7` |
| `onnx/heart_4x_pretrain_fp32_op17.onnx` | `c3b58cce1b18ba65c602f8937df40da75dfe1dbd208d661f48eb85f8380c4369` |
| `onnx/heart_2x_fp32_op17.onnx` | `08ee34f60303ac086c9b50012fdc3d261086d907756640623035cec1b6030102` |
| `heart_arch.py` | `7b7788cdcf2c3acd61f5b601e291ed7cfdc20c06153aa3c9728a3473380c14cf` |

Converted files' sha256 are in `oracle/weights/config.json` (written by `convert_weights.py`).

## Components (isomorphic to upstream)

| Swift (`Sources/HEARTMLX`) | upstream | notes |
|---|---|---|
| `HEART` | `heart_arch.py` `HEART` | `conv_first` → `patch_embed` (affine) → 6 × `RHAG_RIB` → `norm` → `conv_after_body` + stem residual → conv+LeakyReLU(0.01) → [conv 64→256 + PixelShuffle(2)]^n → `conv_last` → crop. NHWC throughout: `(B, L, C)` tokens and `(B, H, W, C)` are the same memory, so every `view`/`transpose` pair vanishes |
| `RHAG_RIB` / `AttenBlocksRIB` / `HAB_RIB` | same names | attention on `i % 2 == 0`, shift on odd attention indices — decided at construction |
| `RIBWindowAttention` | same | 1×1-conv QKV in `(3, heads, c)` order; RIB features `relu(rib_coords @ to_hidden + hidden_b)` → `einsum("nd,hdr->hnr")`, fp32; `q·30^-½`, `pos_q·8^-½`, concat → 38 → zero-pad to `attentionHeadPad` (64; upstream 40); SDPA `scale = 1`; shift = reflect pad 16 each side, partition, crop |
| `ILN` | `arch_util.py` `iLN` | mean/var over EVERY token and channel per sample, eps 1e-4, `scale = std` (the `std_ema` EMA branch is training-only and never in a checkpoint); statistics in fp32 with **two-stage reductions** (see the lesson below) |
| `CAB` / `ChannelAttention` / `Mlp` / `AffineTransform` / `PatchEmbed` | `hat_iln_arch.py` | exact GELU; `AdaptiveAvgPool2d(1)` = mean over H, W; `nn.Sequential` indices kept as `[Module]` with `Identity` placeholders (`cab.{0,2,3}`, `attention.{1,3}`, `upsample.{0,2}`) |
| `Ops.swift` | `F.pad(reflect/replicate)`, `PixelShuffle`, `check_image_size` | reflect padding is a `take` gather (mlx-swift `PadMode` has no reflect); replicate = `edge` |
| `HEART_Playback` | — | `PlaybackTier` over the SHARED `MLXTileProcessor` (mlx-realesrgan-swift); per-tile + per-group cancellation seams |

Weights: `oracle/convert_weights.py` — upstream keys unchanged (748 tensors ×4 / 746 ×2), conv `(O,I,kH,kW)` → `(O,kH,kW,I)`,
nothing else touched; fp32 lane + fp16 lane (conv/linear tensors fp16; i-LN / affine weight+bias and the four RIB
parameters kept fp32 — 220 of 748).

## Gate tooling

`heart-smoke` (Sources/Smoke; `xcrun swift build -c release --target HEARTSmoke`, binary at
`.build/out/Products/Release/heart-smoke`, AB-L-0171):

```
heart-smoke keys <weightsDir> [fp32|fp16]                                   # S0
heart-smoke gate <weightsDir> <goldensDir> [--gpu] [--head 40|64] [--fp16] [--only taps,e2e] [--sizes 128,512]   # S1/S2
heart-smoke run <in.png> <out.png> <weightsDir> [--variant V] [--fp32] [--cpu] [--head N] [--tile N] [--overlap N] [--whole N]
heart-smoke engine <in.png> <out.png> <weightsDir> [--variant V] [--fp32] [--scale N] [--raw]                     # S7
heart-smoke perf <weightsDir> [--sizes 256,512] [--rounds 5] [--arms fp32-64,fp32-40,fp16-64,fp16-40] [--json f]  # H4
heart-smoke tiles <weightsDir> <benchDir> <outDir> [--tiles 256,384,512] [--overlaps 32,64] [--memory-only]        # H5
heart-smoke cancel <weightsDir> [--after 1.5]                                                                    # CAN live
```

Goldens = `oracle/dump_goldens.py` output (torch CPU fp32, seeded uniform inputs): five tap files on the `.fidelity`
checkpoint (64², 96², 100×140 = the `check_image_size` reflect path → 128×160, 20×20 = reflect with pad 12 < 20, 12×12 = the
**replicate** fallback with pad 20 ≥ 12 — the plan's "20×20 replicate" case is a reflect case by upstream's rule), 63 arrays
each: the tables (`rib_coords`, every attention block's `q_pos`/`k_pos`, the reflect-shift tensor), the stem, every sub-op of
block g0b0, block g0b1 (no attention), block g0b2 (shifted), RHAG 0, the chained g5b4 attention, `features`, the tail, the
output; plus `e2e_<ckpt>_{128,512}` with `in`/`out`/`onnx` for every checkpoint. Gate metric = max|Δ| relative to the golden's
|max| (`rel`) plus cosine; thresholds 2e-6 primitives · 1e-5 sub-ops · 5e-5 chained · 2e-4 e2e (PSNR reported). The CPU fp32
lane is the strict one; `--gpu` reports the Metal gap (TF32 GEMMs on M5 — set `MLX_ENABLE_TF32=0`, AB-L-0175).

## Phase gates (stamp AFTER the run — a row is `pending` until a tool result says otherwise)

| phase | gate | status |
|---|---|---|
| H0 / S0 key contract | `heart-smoke keys`: every converted file 0 missing / 0 unused, shapes load, 748 (×4) / 746 (×2) tensors; `rib_coords` and `std_ema` are not module parameters | **PASSED 2026-09-28** — 8/8 files (fp32: 748 fp32 tensors; fp16 lane: 528 fp16 + 220 fp32), 16,677,399 / 16,529,687 params; scale derived from the `upsample` conv count (`oracle/reports/s0-keys.log`) |
| H1 / S1 tables | `rib_coords` at tolerance 0 → **≤ 1 ulp** (see lesson 1); 36 `q_pos`/`k_pos` tables rel ≤ 2e-6; the reflect-shift tensor at tolerance 0 | **PASSED 2026-09-28** — `rib_coords` max 5.96e-8 (2272/43008 entries differ by exactly one ulp), position features rel ≤ 1.9e-7, `g0b2.shiftpad` on the oracle's own input **0.0** at all five sizes (`oracle/reports/s1-taps.log`) |
| H1 / S1 sub-ops | per sub-op at 64², 96², 100×140, 20×20, 12×12: `padded` exact; stem 2e-6; norm1/CAB/attention/norm2/MLP/block/RHAG 1e-5; chained taps 5e-5; module e2e 2e-4 | **PASSED 2026-09-28** — 173/173 rungs; worst sub-op rel 9.6e-6 (`g0b0.mlp` at 12×12), chained `features`/tail ≤ 3.7e-5, e2e PSNR 124.5–127.6 dB at every size, the padded (`check_image_size`) reflect and replicate paths exact. Head-pad probe: 64 vs 40 **bit-identical** (max|Δ| 0) |
| H1 probes | `(heads, 3, c)` QKV read must fail loudly; `(r, r, C)` pixel shuffle must fail loudly | **committed** — `HEARTCoreTests.testQKVOrderProbeDiscriminates` (rel > 0.1) and `testPixelShuffleMatchesTorchAndTheWrongOrderingFailsLoudly` |
| H2 / S2 e2e 128² | every variant vs torch ≥ 90 dB; vs the author's ONNX | **PASSED 2026-09-28** — vs torch: `.fidelity` **127.5 dB**, `.sharp` 125.3, `.clean` 122.5, `.clean2x` 124.0; vs ONNX: 90.2 / 84.9 / **7.8** / 95.8 dB = exactly torch's own agreement with those files (`oracle/reports/s2-e2e-128.log`; lesson 2 on the two ONNX caveats) |
| H2 / S2 e2e 512² | same at 512² (→ 2048² / 1024² out) | **PASSED 2026-09-28** — vs torch: `.fidelity` **127.4 dB**, `.sharp` 125.7, `.clean` 122.4, `.clean2x` 124.0 (worst sub-pixel rel 6.0e-6); vs ONNX 54.5 / 48.9 / 7.9 / 59.8 dB — again identical to torch-vs-ONNX at this size (lesson 2b); 66–100 s per forward on the CPU stream (`oracle/reports/s2-e2e-512.log`) |
| S2b GPU eyeball | `run` per variant on Metal, TF32 off and on, on a real image | pending — GPU window |
| H3 dtype | fp16 (i-LN statistics, RIB tables, softmax fp32) vs the fp32 lane on the 27 bench cells + wall time; residual stream fp16 vs fp32 | pending — GPU window |
| H4 perf | idle-GPU bracket, arms interleaved (fp32/fp16 × head 64/40), 256² and 512²; bar 540–590 ms/Mpx at head 64; RealPLKSR anchor 111 ms/Mpx | pending — GPU window |
| H5 tiling + memory + ×2 | whole-frame vs tiled 256/384/512 (overlap 32/64) on the 27 cells: SSIMULACRA2 change ≤ 0.1, dB tiled-vs-whole, peak memory; whole-frame memory at 960×540 and 1920×1080; `heart_2x` vs `.fidelity` ×4 + downsample | pending — GPU window |
| H6 package | C0–C14 + MAT-1..5 + CAN-1..3 + split footprint at five sizes + `BudgetAware` | offline suites written; `swift test` pending; footprints PROVISIONAL |
| H7 publish | `xocialize/mlx-heart-swift` public v0.1.0; `mlx-community/HEART-{fp16,fp32}` + model cards; the VOSR2-style published-artifact check; registry row | pending |

## Lessons (filed on the bridge at close-out)

1. **torch's float32 `sin`/`cos` are not correctly rounded** (Sleef, ≤ 1 ulp). A `Double`-computed coordinate table
   rounded once to float32 differs from `rib_coords` in 2272 of 43008 entries by exactly one ulp (5.96e-8). "Tolerance 0"
   on a table that contains transcendental values is therefore a claim about the *library*, not the port: gate such
   tables at ≤ 1 ulp and gate what consumes them relatively (here the position features land at 1.9e-7).
2. **The second oracle has two caveats.** (a) The author's `heart_4x_pretrain_fp32_op17.onnx` does not reproduce
   `heart_4x_pretrain.safetensors` — torch-vs-ORT **7.8 dB** at 128² and 512² (max|Δ| 1.3), and the file is a different
   size from the other two ×4 exports (72.3 vs 75.0 MB) — it was exported from some other checkpoint; the `.clean` variant
   has torch as its only oracle. (b) torch-vs-ORT agreement degrades with image size on every checkpoint (`.fidelity` 90.2 dB
   at 128² → **54.5 dB** at 512²; `.sharp` 84.9 → 48.9; `.clean2x` 95.8 → 59.8): see lesson 3 — the ONNX bar is 128².
3. **MLX's CPU reductions are naive fp32 accumulations; the GPU's are tree-based.** Measured on one i-LN statistic (a sum
   over every token and channel of the image, `oracle/reports/reduction-probe.log`): std relative error on the CPU stream
   6.3e-7 at 32²·180, 4.7e-6 at 64², 5.8e-5 at 128×160, **2.1e-3 at 512²** (47 M elements); on the GPU 6.7e-9 / 1.2e-8 /
   7.0e-8 / 2.4e-10; torch CPU (double-accumulated Welford) 6.7e-9 … 2.4e-10. The first S1 run read 87 dB at 64² and every
   downstream rung at ~1e-4 for exactly this reason. A two-stage reduction (`chunkedSum`: chunks of ≤ 4096, then across
   chunks) brings the CPU lane to 1.4e-7 at 512² and is neutral on the GPU; after it, 64² reads 127.6 dB. Corollary: any
   whole-tensor normalisation ported to MLX must not gate its parity on the CPU lane with the plain `mean`/`variance` — and
   ORT's ReduceMean is likely the same class of failure (lesson 2b).
4. **`[0..., 0..., 0..., 0 ..< d]` on its own line after a call is parsed as an array literal**, not a subscript — keep
   MLX slices on the call's line or bind the call first.

## Hazards carried in

- **mlx#3797 NAX split-K GEMM** (mlx-swift ≤ 0.31.6): half-precision GEMMs with a large K corrupt in a row window. HEART's
  largest K is 360 (`fc2`) / 540 (`to_qkv` as a matmul) — far below the K ≥ 10240 dispatch window; not applicable.
- i-LN is **not tile-invariant**: a tile is normalised by its own statistics, so tiled output ≠ whole-frame output by
  construction; H5 measures the geometry instead of assuming the siblings' 256/32.
- `check_image_size` pads to /32 with **reflect** only when every non-zero pad is smaller than its axis, else **replicate**
  for both axes (inputs ≤ 16 px on a side); the crop is `[:h·s, :w·s]`.
- **fp32 on M5 is TF32-class** unless the host process sets `MLX_ENABLE_TF32=0` before its first GEMM (AB-L-0175); the fp32
  lane is the parity lane, the shipping lane is fp16 (H3).
- Per-group cancellation seams force an `eval` after each RHAG (six per forward) — measured against a single-eval forward
  in H4.
