//
//  HEART.swift — mlx-heart-swift / HEARTMLX
//
//  HEART — Hybrid Efficient Attention with Rank-factorized bias Transformer (Philip Hofmann / Phips, Apache-2.0),
//  ported 1:1 from `heart_arch.py` (Phips/HEART @ 868878ce4c25) and the traiNNer-redux helpers it imports
//  (`hat_iln_arch.py`: CAB, ChannelAttention, Mlp, AffineTransform, PatchEmbed, Upsample; `arch_util.py`: iLN — both
//  @ 686293190aca, Apache-2.0). HAT (XPixelGroup, arXiv 2205.04437) and HAT-iLN (arXiv 2504.06629) are the
//  architecture family; RIB is SST's rank-factorized implicit bias (arXiv 2603.06738).
//
//  Same class names, same decomposition, same forward order; only PyTorch → MLX op substitutions, and one layout
//  change: everything is NHWC. Upstream's token layout (B, L, C) and NHWC (B, H, W, C) are the same memory, so
//  every `view`/`transpose` pair around `patch_embed` / `patch_unembed` / the attention's channels-first entry
//  disappears rather than becoming a transpose.
//
//  The traps this file encodes (NERVE-HEART-PORT-PLAN §3.3), each with a gate in the smoke CLI / tests:
//   • QKV channel order is `(3, heads, c)` — `[q heads][k heads][v heads]` — not `(heads, 3, c)`.
//   • The shift is a NON-WRAPPING reflect pad-and-partition (16 px each side, one extra window row/col), attend,
//     crop — not Swin's cyclic roll + mask.
//   • Block type is decided at CONSTRUCTION: attention on `i % attention_freq == 0`, shift on odd attention
//     indices within a group (0 plain, 2 shifted, 4 plain per group).
//   • i-LN normalises over EVERY token and channel of the sample; `std_ema` is non-persistent and never in a
//     checkpoint, so at inference the rescale is always the raw per-sample std — the EMA branch is not ported.
//   • Q is pre-scaled by `head_dim^-0.5` and the RIB position query by `rank^-0.5`, so SDPA runs with `scale = 1`.
//   • Exact (erf) GELU; LeakyReLU slope 0.01; `AdaptiveAvgPool2d(1)` is a mean over H and W.
//   • `rib_coords` and the per-block position features are constants, boxed OUTSIDE the `Module` tree
//     (`Module` reflection collects every stored `MLXArray` as a parameter — the SCUNet S0 trap).
//

import Foundation
import MLX
import MLXFast
import MLXNN

/// Errors from the HEART core.
public enum HEARTError: Error, Sendable, CustomStringConvertible {
    case weightsNotFound(String)
    case loadFailed(String)
    /// The checkpoint's parameter set does not cover the module tree exactly — a silently partial load is the
    /// failure mode that produces plausible-but-wrong output, so it is loud.
    case parameterMismatch(missing: [String], extra: [String])
    case unsupportedScale(Int)

    public var description: String {
        switch self {
        case .weightsNotFound(let p): return "HEART weights not found: \(p)"
        case .loadFailed(let d): return "HEART weight load failed: \(d)"
        case .parameterMismatch(let m, let e):
            return "HEART parameter mismatch — missing \(m.count) \(m.prefix(5)) extra \(e.count) \(e.prefix(5))"
        case .unsupportedScale(let s): return "HEART: unsupported scale \(s) (2^n only)"
        }
    }
}

/// The released architecture (`heart()` in `heart_arch.py`, every checkpoint uses the defaults) plus the port's
/// two runtime knobs. `scale` is derived from the checkpoint by the loader (`upsample` conv count), never trusted
/// from a flag that could disagree with the weights.
public struct HEARTConfig: Sendable, Equatable {
    public var scale: Int = 4
    public var inChannels: Int = 3
    public var embedDim: Int = 180
    public var depths: [Int] = [6, 6, 6, 6, 6, 6]
    public var numHeads: [Int] = [6, 6, 6, 6, 6, 6]
    public var windowSize: Int = 32
    public var compressRatio: Int = 3
    public var squeezeFactor: Int = 30
    public var convScale: Float = 0.01
    public var mlpRatio: Float = 2.0
    public var numFeat: Int = 64
    public var rank: Int = 8
    public var ribHiddenDim: Int = 32
    public var ribNFreqs: Int = 10
    public var attentionFreq: Int = 2
    public var ilnEps: Float = 1e-4

    /// Head dimension the concatenated (q ‖ pos_q) is zero-padded to before SDPA. Upstream pads 38 → 40 (the next
    /// multiple of 8). MLX's fused Metal attention kernel accepts 64 / 80 / 128 only; **64** keeps the math
    /// identical (zero columns contribute exact zeros to every dot product, and the padded value columns are
    /// sliced off) and measured 540–590 ms per output Mpx against 754–763 at 40 (AB-R-0370). The CPU stream
    /// always takes the unfused path, so parity gates are unaffected by this knob.
    public var attentionHeadPad: Int = 64
    /// Keep the residual stream (the block-to-block `x`) in float32 when the weights are half precision — the
    /// AMP-style posture the model was trained under. Measured in PORTING-SPEC H3; the convs, attention and MLP
    /// still run in the weights' dtype.
    public var residualStreamFloat32: Bool = false

    public init() {}
    public init(scale: Int) { self.scale = scale }

    public var headDim: Int { embedDim / numHeads[0] }
    /// 2 + 4·n_freqs — the RIB implicit net's input width.
    public var ribInputDim: Int { 2 + 4 * ribNFreqs }
    /// `int(math.log(scale, 2))` pixel-shuffle stages of the ×2^n upsampler.
    public var upsampleStages: Int {
        precondition(scale > 0 && (scale & (scale - 1)) == 0, "HEART upsampler supports 2^n scales")
        var n = 0, s = scale
        while s > 1 { s >>= 1; n += 1 }
        return n
    }
}

// MARK: - helpers (traiNNer-redux hat_iln_arch.py / arch_util.py)

/// `AffineTransform`: `weight * x + bias` on the channel axis, no normalisation. The `patch_embed.norm` and the
/// final `norm` of an i-LN network.
public final class AffineTransform: Module, UnaryLayer, @unchecked Sendable {
    @ParameterInfo(key: "weight") public var weight: MLXArray
    @ParameterInfo(key: "bias") public var bias: MLXArray

    public init(_ dim: Int) {
        self._weight.wrappedValue = MLXArray.ones([dim])
        self._bias.wrappedValue = MLXArray.zeros([dim])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        (weight.asType(x.dtype) * x + bias.asType(x.dtype))
    }
}

/// i-LN — Image Restoration Transformer Tailored Layer Normalization (HAT-iLN, arXiv 2504.06629).
///
/// Normalises across EVERY spatial position and channel of a sample (not per token), returning the normalised
/// tensor and the per-sample `std` the block rescales its branches by. Upstream's `std_ema` is a non-persistent
/// buffer that is only updated in training and, at zero, leaves the scale as the raw `std` — the checkpoint never
/// carries it, so the inference behaviour ported here is exactly `scale = std`. Statistics run in float32 whatever
/// the activation dtype (the H3 "reductions held at fp32" posture).
public final class ILN: Module, @unchecked Sendable {
    @ParameterInfo(key: "weight") public var weight: MLXArray
    @ParameterInfo(key: "bias") public var bias: MLXArray
    public let eps: Float

    public init(_ dim: Int, eps: Float = 1e-4) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dim])
        self._bias.wrappedValue = MLXArray.zeros([dim])
    }

    /// NHWC in → (normalised NHWC in the input dtype, std as (B, 1, 1, 1) float32).
    public func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let b = x.dim(0)
        let xf = x.asType(.float32)
        // Two-pass, two-stage statistics (see `chunkedSum`): MLX's CPU reductions are naive and the sum here spans
        // the whole image, so the plain `mean`/`variance` drift to 1e-3 relative at 512² on the parity lane.
        let flat = xf.reshaped([b, -1])
        let mean = chunkedMean(lastAxisOf: flat).reshaped([b, 1, 1, 1])
        let centered = xf - mean
        let variance = chunkedMean(lastAxisOf: (centered * centered).reshaped([b, -1])).reshaped([b, 1, 1, 1])   // unbiased=False
        let std = MLX.sqrt(variance + eps)
        let xn = centered / std
        return ((weight * xn + bias).asType(x.dtype), std)
    }
}

/// `ChannelAttention`: global average pool → 1×1 conv (C → C/squeeze) → ReLU → 1×1 conv → sigmoid → gate.
/// Held as `[Module]` so the parameter-free stages keep upstream's `nn.Sequential` indices (`attention.1`,
/// `attention.3`).
public final class ChannelAttention: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "attention") public var attention: [Module]

    public init(numFeat: Int, squeezeFactor: Int) {
        self._attention.wrappedValue = [
            Identity(),                                                                       // AdaptiveAvgPool2d(1)
            Conv2d(inputChannels: numFeat, outputChannels: numFeat / squeezeFactor, kernelSize: 1, padding: 0),
            Identity(),                                                                       // ReLU
            Conv2d(inputChannels: numFeat / squeezeFactor, outputChannels: numFeat, kernelSize: 1, padding: 0),
        ]
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let c1 = attention[1] as! Conv2d, c3 = attention[3] as! Conv2d
        // AdaptiveAvgPool2d(1) = the mean over H and W — two-stage over the spatial axis (see `chunkedSum`).
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let pooled = chunkedMean(lastAxisOf: x.reshaped([b, h * w, c]).transposed(0, 2, 1)).reshaped([b, 1, 1, c])
        let y = sigmoid(c3(relu(c1(pooled))))
        return x * y
    }
}

/// `CAB`: conv3×3 (C → C/3) → GELU → conv3×3 (C/3 → C) → ChannelAttention. `cab.{0,2,3}` are the checkpoint keys.
public final class CAB: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "cab") public var cab: [Module]

    public init(numFeat: Int, compressRatio: Int, squeezeFactor: Int) {
        self._cab.wrappedValue = [
            Conv2d(inputChannels: numFeat, outputChannels: numFeat / compressRatio, kernelSize: 3, padding: 1),
            Identity(),                                                                       // nn.GELU() — exact erf
            Conv2d(inputChannels: numFeat / compressRatio, outputChannels: numFeat, kernelSize: 3, padding: 1),
            ChannelAttention(numFeat: numFeat, squeezeFactor: squeezeFactor),
        ]
    }

    /// The conv branch before channel attention (tapped by the S1 gate).
    public func preAttention(_ x: MLXArray) -> MLXArray {
        let c0 = cab[0] as! Conv2d, c2 = cab[2] as! Conv2d
        return c2(gelu(c0(x)))
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        (cab[3] as! ChannelAttention)(preAttention(x))
    }
}

/// `Mlp`: fc1 → GELU (exact) → fc2; dropout is identity at inference.
public final class Mlp: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "fc1") public var fc1: Linear
    @ModuleInfo(key: "fc2") public var fc2: Linear

    public init(inFeatures: Int, hiddenFeatures: Int) {
        self._fc1.wrappedValue = Linear(inFeatures, hiddenFeatures, bias: true)
        self._fc2.wrappedValue = Linear(hiddenFeatures, inFeatures, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// `PatchEmbed` with `norm_layer=AffineTransform`: flatten is a no-op in NHWC, so only the affine remains.
public final class PatchEmbed: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "norm") public var norm: AffineTransform
    public init(embedDim: Int) { self._norm.wrappedValue = AffineTransform(embedDim) }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { norm(x) }
}

// MARK: - RIB window attention

/// The normalised window coordinates with their Fourier features — `rib_coords`, a non-persistent buffer upstream
/// rebuilt from `window_size` and `n_freqs`, shared by every attention block. Boxed outside the `Module` tree.
///
/// Built in `Double` and rounded once to float32. ⚠️ That is *correctly rounded*, which torch's float32 `sin`/`cos`
/// (Sleef, 1-ulp) are not: measured 2272 of 43008 entries differ from the oracle's table by exactly one ulp
/// (5.96e-8), so the table gate is "≤ 1 ulp", and the downstream position features are gated relatively.
public final class RIBCoordinates: @unchecked Sendable {
    public let coords: MLXArray      // (window², 2 + 4·nFreqs) float32
    public let window: Int
    public let nFreqs: Int

    private static let lock = NSLock()
    private nonisolated(unsafe) static var cache: [String: RIBCoordinates] = [:]

    public static func shared(window: Int, nFreqs: Int) -> RIBCoordinates {
        lock.lock(); defer { lock.unlock() }
        let key = "\(window)/\(nFreqs)"
        if let t = cache[key] { return t }
        let t = RIBCoordinates(window: window, nFreqs: nFreqs)
        cache[key] = t
        return t
    }

    public static func table(window: Int, nFreqs: Int) -> [Float] {
        let n = window * window, width = 2 + 4 * nFreqs
        var out = [Float](repeating: 0, count: n * width)
        for p in 0 ..< n {
            // torch.meshgrid(arange(wh), arange(ww), indexing="ij") → (yy, xx); coords = stack([xx, yy]) → (x, y)
            let yy = p / window, xx = p % window
            // (2·(c + 0.5) / w) − 1, evaluated in float32 exactly as torch does (all operands are exact binary fractions)
            let x = Float(2.0 * (Float(xx) + 0.5) / Float(window)) - 1.0
            let y = Float(2.0 * (Float(yy) + 0.5) / Float(window)) - 1.0
            out[p * width + 0] = x
            out[p * width + 1] = y
            for i in 0 ..< nFreqs {
                let f = Float(1 << i)
                let ax = Double(x * f), ay = Double(y * f)      // the float32 products are exact (powers of two)
                let base = 2 + 4 * i
                out[p * width + base + 0] = Float(sin(ax))
                out[p * width + base + 1] = Float(sin(ay))
                out[p * width + base + 2] = Float(cos(ax))
                out[p * width + base + 3] = Float(cos(ay))
            }
        }
        return out
    }

    private init(window: Int, nFreqs: Int) {
        self.window = window
        self.nFreqs = nFreqs
        self.coords = MLXArray(Self.table(window: window, nFreqs: nFreqs), [window * window, 2 + 4 * nFreqs])
    }
}

/// A block's precomputed attention-side constants: the RIB position features already scaled (`pos_q · rank^-0.5`)
/// and zero-padded to `headPad − headDim` columns so the (q ‖ pos_q ‖ 0) concatenation is a single op per call,
/// and the matching all-zero value padding. Depends only on the block's weights → computed once after load.
final class RIBFeatures: @unchecked Sendable {
    let qExtra: MLXArray      // (1, heads, N, headPad − headDim)
    let kExtra: MLXArray
    let vExtra: MLXArray      // zeros
    let qPos: MLXArray        // (heads, N, rank) float32, unscaled — the S1 table gate reads these
    let kPos: MLXArray
    let dtype: DType

    init(qPos: MLXArray, kPos: MLXArray, rank: Int, headDim: Int, headPad: Int, dtype: DType) {
        let heads = qPos.dim(0), n = qPos.dim(1)
        let extra = headPad - headDim
        precondition(extra >= rank, "attentionHeadPad \(headPad) must leave room for headDim \(headDim) + rank \(rank)")
        let zeros = MLXArray.zeros([1, heads, n, extra - rank], dtype: .float32)
        let qs = (qPos * pow(Float(rank), -0.5)).expandedDimensions(axis: 0)
        let ks = kPos.expandedDimensions(axis: 0)
        self.qExtra = concatenated([qs, zeros], axis: -1).asType(dtype)
        self.kExtra = concatenated([ks, zeros], axis: -1).asType(dtype)
        self.vExtra = MLXArray.zeros([1, heads, n, extra], dtype: dtype)
        self.qPos = qPos
        self.kPos = kPos
        self.dtype = dtype
        eval(self.qExtra, self.kExtra, self.vExtra)
    }
}

/// `RIBWindowAttention`: window attention whose relative-position bias is a low-rank dot product of implicit
/// position features concatenated onto Q and K (SST, arXiv 2603.06738) — no bias table, no mask, fused SDPA.
/// Shifted windows use non-wrapping reflect pad-and-partition. NHWC in, NHWC out; `window`-aligned inputs.
public final class RIBWindowAttention: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "to_qkv") public var toQKV: Conv2d
    @ModuleInfo(key: "to_out") public var toOut: Conv2d
    @ParameterInfo(key: "to_hidden") public var toHidden: MLXArray   // (n_input, hidden)
    @ParameterInfo(key: "hidden_b") public var hiddenB: MLXArray     // (1, hidden)
    @ParameterInfo(key: "to_q") public var toQ: MLXArray             // (heads, hidden, rank)
    @ParameterInfo(key: "to_k") public var toK: MLXArray

    public let window: Int
    public let numHeads: Int
    public let headDim: Int
    public let rank: Int
    public let shift: Bool
    public let headPad: Int
    private let coordinates: RIBCoordinates
    /// Boxed, lazily built from the loaded weights; dropped by `invalidate()` when weights change.
    private var features: RIBFeatures?
    private let featuresLock = NSLock()

    public init(dim: Int, windowSize: Int, numHeads: Int, rank: Int, ribHiddenDim: Int, ribNFreqs: Int,
                shift: Bool, headPad: Int) {
        precondition(dim % numHeads == 0, "dim must be divisible by num_heads")
        self.window = windowSize
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.rank = rank
        self.shift = shift
        self.headPad = headPad
        self.coordinates = RIBCoordinates.shared(window: windowSize, nFreqs: ribNFreqs)
        let nInput = 2 + 4 * ribNFreqs
        self._toQKV.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim * 3, kernelSize: 1, padding: 0)
        self._toOut.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1, padding: 0)
        self._toHidden.wrappedValue = MLXArray.zeros([nInput, ribHiddenDim])
        self._hiddenB.wrappedValue = MLXArray.zeros([1, ribHiddenDim])
        self._toQ.wrappedValue = MLXArray.zeros([numHeads, ribHiddenDim, rank])
        self._toK.wrappedValue = MLXArray.zeros([numHeads, ribHiddenDim, rank])
    }

    /// The position features exactly as upstream computes them per forward (fp32 regardless of the model dtype):
    /// `intermediate = relu(rib_coords @ to_hidden + hidden_b)`, `pos = einsum("nd,hdr->hnr", intermediate, to_q/k)`.
    public func positionFeatures() -> (qPos: MLXArray, kPos: MLXArray) {
        let inter = relu(matmul(coordinates.coords, toHidden.asType(.float32)) + hiddenB.asType(.float32))  // (N, hidden)
        let q = matmul(inter.expandedDimensions(axis: 0), toQ.asType(.float32))   // (heads, N, rank)
        let k = matmul(inter.expandedDimensions(axis: 0), toK.asType(.float32))
        return (q, k)
    }

    func invalidate() { featuresLock.lock(); features = nil; featuresLock.unlock() }

    func prepared(dtype: DType) -> RIBFeatures {
        featuresLock.lock(); defer { featuresLock.unlock() }
        if let f = features, f.dtype == dtype { return f }
        let (q, k) = positionFeatures()
        let f = RIBFeatures(qPos: q, kPos: k, rank: rank, headDim: headDim, headPad: headPad, dtype: dtype)
        features = f
        return f
    }

    /// The reflect-padded tensor a shifted block attends over (tapped by the S1 table gate).
    public func shiftPadded(_ x: MLXArray) -> MLXArray {
        let s = window / 2
        return reflectPad(x, top: s, bottom: s, left: s, right: s)
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        let (b, h, w, c) = (input.dim(0), input.dim(1), input.dim(2), input.dim(3))
        // `pad_to_win` is a no-op here: HEART.check_image_size owns the window alignment (and the replicate fallback).
        precondition(h % window == 0 && w % window == 0, "attention input must be window-aligned (\(h)×\(w))")
        var x = input
        let s = shift ? window / 2 : 0
        if shift { x = shiftPadded(x) }          // one extra window row and column; alignment is kept
        let hp = x.dim(1), wp = x.dim(2)
        let nh = hp / window, nw = wp / window
        let n = window * window
        let nWin = b * nh * nw

        // "b (qkv heads c) (h wh) (w ww) -> qkv (b h w) (wh ww) heads c": the 3·dim projection channels are
        // [q heads][k heads][v heads]. (heads, 3, c) is shape-identical and wrong — probed in the tests.
        let qkv = toQKV(x)
            .reshaped([b, nh, window, nw, window, 3, numHeads, headDim])
            .transposed(5, 0, 1, 3, 6, 2, 4, 7)                 // (3, b, nh, nw, heads, wh, ww, c)
            .reshaped([3, nWin, numHeads, n, headDim])          // (3, Bwin, heads, N, c) — SDPA layout
        let q = qkv[0] * pow(Float(headDim), -0.5)              // pre-scaled (SST eq. 5) → SDPA scale = 1
        let k = qkv[1]
        let v = qkv[2]

        let f = prepared(dtype: q.dtype)
        let extra = headPad - headDim
        let qc = concatenated([q, broadcast(f.qExtra, to: [nWin, numHeads, n, extra])], axis: -1)
        let kc = concatenated([k, broadcast(f.kExtra, to: [nWin, numHeads, n, extra])], axis: -1)
        let vc = concatenated([v, broadcast(f.vExtra, to: [nWin, numHeads, n, extra])], axis: -1)
        let attended = MLXFast.scaledDotProductAttention(queries: qc, keys: kc, values: vc, scale: 1.0, mask: nil)
        let o = attended[0..., 0..., 0..., 0 ..< headDim]           // (Bwin, heads, N, c) — the padded columns drop

        // "(b h w) (wh ww) heads c -> b (heads c) (h wh) (w ww)"
        var y = o.reshaped([b, nh, nw, numHeads, window, window, headDim])
            .transposed(0, 1, 4, 2, 5, 3, 6)                    // (b, nh, wh, nw, ww, heads, c)
            .reshaped([b, hp, wp, c])
        if shift { y = y[0..., s ..< (s + h), s ..< (s + w), 0...] }
        return toOut(y)
    }
}

// MARK: - blocks

/// `HAB_RIB`: i-LN → (CAB · conv_scale + RIB window attention) · std → residual; i-LN → MLP · std → residual.
public final class HAB_RIB: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "norm1") public var norm1: ILN
    @ModuleInfo(key: "attn") public var attn: RIBWindowAttention?
    @ModuleInfo(key: "conv_block") public var convBlock: CAB
    @ModuleInfo(key: "norm2") public var norm2: ILN
    @ModuleInfo(key: "mlp") public var mlp: Mlp

    public let convScale: Float
    public let useAttention: Bool
    public let shift: Bool
    /// See `HEARTConfig.residualStreamFloat32`; set by the owning `HEART`.
    var residualStreamFloat32 = false

    public init(config c: HEARTConfig, group: Int, shift: Bool, useAttention: Bool) {
        self.convScale = c.convScale
        self.useAttention = useAttention
        self.shift = shift
        self._norm1.wrappedValue = ILN(c.embedDim, eps: c.ilnEps)
        self._attn.wrappedValue = useAttention
            ? RIBWindowAttention(dim: c.embedDim, windowSize: c.windowSize, numHeads: c.numHeads[group], rank: c.rank,
                                 ribHiddenDim: c.ribHiddenDim, ribNFreqs: c.ribNFreqs, shift: shift,
                                 headPad: c.attentionHeadPad)
            : nil
        self._convBlock.wrappedValue = CAB(numFeat: c.embedDim, compressRatio: c.compressRatio,
                                           squeezeFactor: c.squeezeFactor)
        self._norm2.wrappedValue = ILN(c.embedDim, eps: c.ilnEps)
        self._mlp.wrappedValue = Mlp(inFeatures: c.embedDim, hiddenFeatures: Int(Float(c.embedDim) * c.mlpRatio))
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let streamDtype: DType = residualStreamFloat32 ? .float32 : x.dtype
        let shortcut = x.asType(streamDtype)
        let (xn, std1) = norm1(x)
        var residual = convBlock(xn) * convScale
        if let attn { residual = residual + attn(xn) }
        var y = shortcut + std1.asType(streamDtype) * residual.asType(streamDtype)
        let (xn2, std2) = norm2(y)
        y = y + std2.asType(streamDtype) * mlp(xn2).asType(streamDtype)
        return y
    }
}

/// `AttenBlocksRIB`: the block stack; `use_attention = i % attention_freq == 0`, `shift = attn_idx % 2 == 1`.
public final class AttenBlocksRIB: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "blocks") public var blocks: [HAB_RIB]

    public init(config c: HEARTConfig, group: Int) {
        var blocks: [HAB_RIB] = []
        var attnIdx = 0
        for i in 0 ..< c.depths[group] {
            let useAttention = i % c.attentionFreq == 0
            blocks.append(HAB_RIB(config: c, group: group,
                                  shift: useAttention ? (attnIdx % 2 == 1) : false,
                                  useAttention: useAttention))
            if useAttention { attnIdx += 1 }
        }
        self._blocks.wrappedValue = blocks
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for blk in blocks { y = blk(y) }
        return y
    }
}

/// `RHAG_RIB`: Residual Hybrid Attention Group — blocks → conv3×3 → + input. (`patch_embed` / `patch_unembed`
/// inside the group carry no parameters and are the identity in NHWC.)
public final class RHAG_RIB: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "residual_group") public var residualGroup: AttenBlocksRIB
    @ModuleInfo(key: "conv") public var conv: Conv2d

    public init(config c: HEARTConfig, group: Int) {
        self._residualGroup.wrappedValue = AttenBlocksRIB(config: c, group: group)
        self._conv.wrappedValue = Conv2d(inputChannels: c.embedDim, outputChannels: c.embedDim, kernelSize: 3, padding: 1)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(residualGroup(x)) + x
    }
}

// MARK: - the network

/// HEART. NHWC float in [0, 1] → NHWC at `scale`×. Born in inference mode (C14).
public final class HEART: Module, @unchecked Sendable {
    @ModuleInfo(key: "conv_first") public var convFirst: Conv2d
    @ModuleInfo(key: "patch_embed") public var patchEmbed: PatchEmbed
    @ModuleInfo(key: "layers") public var layers: [RHAG_RIB]
    @ModuleInfo(key: "norm") public var norm: AffineTransform
    @ModuleInfo(key: "conv_after_body") public var convAfterBody: Conv2d
    /// `[Conv2d]` — the LeakyReLU(0.01) at upstream index 1 has no parameters and is applied in code.
    @ModuleInfo(key: "conv_before_upsample") public var convBeforeUpsample: [Module]
    /// `[Conv2d, PixelShuffle, Conv2d, …]` with the shuffles as `Identity` placeholders so the checkpoint's
    /// `upsample.{0,2}` indices line up; the trailing shuffle is omitted (a trailing gap would not round-trip).
    @ModuleInfo(key: "upsample") public var upsample: [Module]
    @ModuleInfo(key: "conv_last") public var convLast: Conv2d

    public let config: HEARTConfig
    /// Parameter tensor count of the released ×4 / ×2 checkpoints — the S0 contract.
    public static let tensorCountX4 = 748
    public static let tensorCountX2 = 746
    public static let parameterCountX4 = 16_677_399
    public static let parameterCountX2 = 16_529_687

    public init(config: HEARTConfig = HEARTConfig()) {
        self.config = config
        let c = config
        self._convFirst.wrappedValue = Conv2d(inputChannels: c.inChannels, outputChannels: c.embedDim, kernelSize: 3, padding: 1)
        self._patchEmbed.wrappedValue = PatchEmbed(embedDim: c.embedDim)
        self._layers.wrappedValue = (0 ..< c.depths.count).map { RHAG_RIB(config: c, group: $0) }
        self._norm.wrappedValue = AffineTransform(c.embedDim)
        self._convAfterBody.wrappedValue = Conv2d(inputChannels: c.embedDim, outputChannels: c.embedDim, kernelSize: 3, padding: 1)
        self._convBeforeUpsample.wrappedValue = [
            Conv2d(inputChannels: c.embedDim, outputChannels: c.numFeat, kernelSize: 3, padding: 1),
        ]
        var up: [Module] = []
        for stage in 0 ..< c.upsampleStages {
            if stage > 0 { up.append(Identity()) }                                     // nn.PixelShuffle(2)
            up.append(Conv2d(inputChannels: c.numFeat, outputChannels: 4 * c.numFeat, kernelSize: 3, padding: 1))
        }
        self._upsample.wrappedValue = up
        self._convLast.wrappedValue = Conv2d(inputChannels: c.numFeat, outputChannels: c.inChannels, kernelSize: 3, padding: 1)
        super.init()
        // C14: the single construction choke point every load path funnels through.
        train(false)
        for layer in layers { for blk in layer.residualGroup.blocks { blk.residualStreamFloat32 = c.residualStreamFloat32 } }
    }

    /// The dtype the loaded weights run in (`.float32` until weights are loaded).
    public var computeDtype: DType { convFirst.weight.dtype }
    public var scale: Int { config.scale }

    /// Every attention block in forward order (18 for the released config).
    public var attentionBlocks: [RIBWindowAttention] {
        layers.flatMap { $0.residualGroup.blocks.compactMap { $0.attn } }
    }

    // MARK: forward

    /// `conv_first` + `patch_embed` on an already window-aligned NHWC input: returns (stem features `f`, tokens).
    public func stem(_ x: MLXArray) -> (f: MLXArray, y: MLXArray) {
        let f = convFirst(x)
        return (f, patchEmbed(f))
    }

    /// `norm` → `conv_after_body` + stem residual → `conv_before_upsample` → upsample → `conv_last`.
    public func tail(_ y: MLXArray, stemFeatures f: MLXArray) -> MLXArray {
        var z = convAfterBody(norm(y)) + f
        z = leakyRelu((convBeforeUpsample[0] as! Conv2d)(z), negativeSlope: 0.01)
        for stage in 0 ..< config.upsampleStages {
            z = pixelShuffleNHWC((upsample[2 * stage] as! Conv2d)(z), 2)
        }
        return convLast(z)
    }

    /// The whole forward, uninterrupted.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        try! forward(x)
    }

    /// The forward with a cooperative seam after every residual group: `checkpoint` (CAN — rethrown unchanged) and
    /// `onGroup(done, total)` (`RunProgress`). When either is supplied the group output is evaluated first, so a
    /// cancel lands within one group's work (~1/6 of the forward) instead of after the whole frame.
    public func forward(_ input: MLXArray,
                        checkpoint: (() throws -> Void)? = nil,
                        onGroup: ((Int, Int) -> Void)? = nil) throws -> MLXArray {
        let h = input.dim(1), w = input.dim(2)
        let x = checkImageSize(input.asType(computeDtype), window: config.windowSize)
        let (f, tokens) = stem(x)
        var y = tokens
        let seam = checkpoint != nil || onGroup != nil
        for (i, layer) in layers.enumerated() {
            y = layer(y)
            if seam {
                eval(y)
                try checkpoint?()
                onGroup?(i + 1, layers.count)
            }
        }
        let out = tail(y, stemFeatures: f)
        return out[0..., 0 ..< (h * config.scale), 0 ..< (w * config.scale), 0...]
    }

    // MARK: weights

    /// Load a converted checkpoint (`oracle/convert_weights.py` layout: upstream keys, conv weights OHWI).
    public func loadWeights(from url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else { throw HEARTError.weightsNotFound(url.path) }
        let arrays: [String: MLXArray]
        do { arrays = try MLX.loadArrays(url: url) } catch { throw HEARTError.loadFailed(String(describing: error)) }
        try loadWeights(arrays)
    }

    /// Strict load: the checkpoint keys must equal the module tree's flattened keys (0 missing / 0 unused) and
    /// every shape must match, else nothing is applied. Weights are materialised on the CPU stream here (the
    /// `loadArrays` stream), then the RIB position tables are rebuilt from them.
    public func loadWeights(_ sd: [String: MLXArray]) throws {
        let expected = Set(parameters().flattened().map(\.0))
        let got = Set(sd.keys)
        let missing = expected.subtracting(got).sorted(), extra = got.subtracting(expected).sorted()
        guard missing.isEmpty && extra.isEmpty else {
            throw HEARTError.parameterMismatch(missing: missing, extra: extra)
        }
        do {
            try update(parameters: ModuleParameters.unflattened(sd), verify: .all)
        } catch {
            throw HEARTError.loadFailed(String(describing: error))
        }
        eval(self)
        for a in attentionBlocks { a.invalidate() }
    }

    /// Derive the checkpoint's scale from its `upsample` conv count — never from a flag that can disagree with it.
    public static func scale(ofCheckpointKeys keys: some Collection<String>) -> Int {
        let convs = keys.filter { $0.hasPrefix("upsample.") && $0.hasSuffix(".weight") }.count
        return 1 << convs
    }

    /// The flattened parameter keys the released architecture expects (the S0 contract, weight-free).
    public static func expectedKeys(scale: Int) -> Set<String> {
        Set(HEART(config: HEARTConfig(scale: scale)).parameters().flattened().map(\.0))
    }
}
