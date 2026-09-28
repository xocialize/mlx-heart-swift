//
//  Ops.swift — mlx-heart-swift / HEARTMLX
//
//  The op substitutions the port needs that MLX-Swift does not ship: PyTorch `reflect` padding (the fleet's
//  mlx-swift `PadMode` is `constant` / `edge` only), `replicate` padding spelled out, a channels-last pixel
//  shuffle, and HEART's `check_image_size`. Everything here is NHWC.
//

import Foundation
import MLX

/// Gather indices that realise PyTorch `F.pad(mode="reflect")` along one axis of length `n`: the edge sample
/// is NOT repeated — position `-k` reads `x[k]`, position `n-1+k` reads `x[n-1-k]`.
///
/// Torch requires `lo < n` and `hi < n`; HEART's `check_image_size` chooses `replicate` when that would not hold.
func reflectIndices(n: Int, lo: Int, hi: Int) -> [Int32] {
    precondition(lo < n && hi < n, "reflect pad needs pad < size (n \(n), lo \(lo), hi \(hi))")
    var idx = [Int32](); idx.reserveCapacity(lo + n + hi)
    if lo > 0 { for k in stride(from: lo, through: 1, by: -1) { idx.append(Int32(k)) } }
    for k in 0 ..< n { idx.append(Int32(k)) }
    if hi > 0 { for k in 1 ... hi { idx.append(Int32(n - 1 - k)) } }
    return idx
}

/// `F.pad(x, (left, right, top, bottom), mode="reflect")` on an NHWC tensor — one gather per padded axis, so the
/// result is bit-identical to torch (it moves samples, it never computes). Zero pads are no-ops.
public func reflectPad(_ x: MLXArray, top: Int, bottom: Int, left: Int, right: Int) -> MLXArray {
    var y = x
    if top > 0 || bottom > 0 {
        y = y.take(MLXArray(reflectIndices(n: y.dim(1), lo: top, hi: bottom)), axis: 1)
    }
    if left > 0 || right > 0 {
        y = y.take(MLXArray(reflectIndices(n: y.dim(2), lo: left, hi: right)), axis: 2)
    }
    return y
}

/// `F.pad(..., mode="replicate")` on an NHWC tensor (MLX `edge`).
public func replicatePad(_ x: MLXArray, top: Int, bottom: Int, left: Int, right: Int) -> MLXArray {
    if top == 0 && bottom == 0 && left == 0 && right == 0 { return x }
    return padded(x, widths: [IntOrPair([0, 0]), IntOrPair([top, bottom]), IntOrPair([left, right]), IntOrPair([0, 0])],
                  mode: .edge)
}

/// NHWC pixel shuffle matching `torch.nn.PixelShuffle` on NCHW.
///
/// Torch views (B, C·r·r, H, W) as (B, C, r1, r2, H, W) — channel index `c·r·r + i·r + j` maps to (c, i, j) —
/// then permutes to (B, C, H, r1, W, r2). Channel-last: (B,H,W,C·r·r) → (B,H,W,C,r1,r2) → (B,H,r1,W,r2,C).
/// ⚠️ Reading the channel as (r1, r2, C) instead is shape-identical and scrambles the image; the core tests keep
/// that probe so the failure stays loud.
public func pixelShuffleNHWC(_ x: MLXArray, _ r: Int) -> MLXArray {
    let (b, h, w, crr) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let c = crr / (r * r)
    return x.reshaped([b, h, w, c, r, r])
        .transposed(0, 1, 4, 2, 5, 3)      // B, H, r1, W, r2, C
        .reshaped([b, h * r, w * r, c])
}

/// HEART's `check_image_size`: pad the bottom/right to a multiple of `window` — `reflect` when every non-zero pad
/// is smaller than its axis, otherwise `replicate` for BOTH axes (upstream picks one mode for the whole call).
public func checkImageSize(_ x: MLXArray, window: Int) -> MLXArray {
    let h = x.dim(1), w = x.dim(2)
    let padH = (window - h % window) % window
    let padW = (window - w % window) % window
    if padH == 0 && padW == 0 { return x }
    let canReflect = (padH == 0 || padH < h) && (padW == 0 || padW < w)
    return canReflect ? reflectPad(x, top: 0, bottom: padH, left: 0, right: padW)
                      : replicatePad(x, top: 0, bottom: padH, left: 0, right: padW)
}

// MARK: - accurate reductions

/// Sum over the last axis in two stages (chunks of up to 4096, then across chunks).
///
/// ⚠️ MLX's CPU `sum` / `mean` / `variance` accumulate naively in float32, so their error grows with the element
/// count — measured on the i-LN statistics (one sum over EVERY token and channel of the image): the std's relative
/// error is 4.7e-6 at 64², 5.8e-5 at 128×160 and **2.1e-3 at 512²** (47 M elements), against 2.4e-10 for the GPU's
/// tree reduction and torch's double-accumulated Welford. Two stages bring the CPU lane to 1.4e-7 at 512² and are
/// exact-neutral on the GPU. Any whole-tensor normalisation gated on the CPU parity lane needs this.
public func chunkedSum(lastAxisOf x: MLXArray) -> MLXArray {
    let n = x.dim(-1)
    var chunk = 1
    for c in [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2] where n % c == 0 { chunk = c; break }
    if chunk == 1 { return x.sum(axis: -1) }
    var shape = x.shape; shape.removeLast(); shape += [n / chunk, chunk]
    return x.reshaped(shape).sum(axis: -1).sum(axis: -1)
}

/// Mean over the last axis, two-stage.
public func chunkedMean(lastAxisOf x: MLXArray) -> MLXArray {
    chunkedSum(lastAxisOf: x) / Float(x.dim(-1))
}
