//
//  HEART_Playback.swift — mlx-heart-swift / HEARTMLX
//
//  A `PlaybackTier` over HEART: model + the SHARED tile driver (RealESRGANMLX.MLXTileProcessor), the same pairing
//  the Real-ESRGAN and RealPLKSR tiers use — so the backends are drop-in siblings behind one call and share the
//  tile compositor's fidelity tests.
//
//  ⚠️ Unlike its conv-only siblings, HEART is NOT tile-invariant: i-LN normalises each block's input over the
//  whole tensor it sees, so a tile is normalised by the tile's statistics and the tiled result differs from the
//  whole-frame result (bit-for-bit it cannot match). PORTING-SPEC H5 measured the tile geometry against
//  whole-frame output on the bench (SSIMULACRA2 vs the reference, PSNR tiled-vs-whole, seam concentration) and
//  the defaults below come from that — they are not the siblings' defaults carried over.
//

import CoreVideo
import Foundation
import MLX
import MLXNN
import RealESRGANMLX

public final class HEART_Playback: PlaybackTier, @unchecked Sendable {

    /// The released checkpoints (Phips/HEART @ 868878ce4c25). File stems follow `oracle/convert_weights.py`.
    public enum Variant: String, Sendable, CaseIterable {
        /// `heart_4x_otf_v2` — 4× OTF fidelity: the best damaged-input arm on the bench, both oracles agreeing. Default.
        case fidelity
        /// `heart_4x_otf_gan` — 4× OTF GAN: RealPLKSR's texture level.
        case sharp
        /// `heart_4x_pretrain` — 4× official pretrain: clean sources.
        case clean
        /// `heart_2x` — 2× official pretrain: the only native ×2 checkpoint.
        case clean2x

        public var checkpointStem: String {
            switch self {
            case .fidelity: return "heart_4x_otf_v2"
            case .sharp: return "heart_4x_otf_gan"
            case .clean: return "heart_4x_pretrain"
            case .clean2x: return "heart_2x"
            }
        }
        public var scale: Int { self == .clean2x ? 2 : 4 }
        public var tierName: String { "heart-\(rawValue)-x\(scale)" }
        /// The converted file name for a precision lane.
        public func fileName(precision: Precision) -> String { "\(checkpointStem)_\(precision.rawValue).safetensors" }
    }

    /// The precision lane = which converted file loads. `fp16` is the shipping lane (i-LN statistics, the RIB
    /// tables and the softmax stay fp32 inside), `fp32` the parity / reference lane.
    public enum Precision: String, Sendable, CaseIterable {
        case fp32, fp16
    }

    // MARK: - PlaybackTier surface

    public let name: String
    public let scaleFactor: Int
    public let inputTileSize: Int
    public let tileOverlap: Int
    public var inputResolution: (width: Int, height: Int) { (inputTileSize, inputTileSize) }
    public var outputResolution: (width: Int, height: Int) {
        (inputTileSize * scaleFactor, inputTileSize * scaleFactor)
    }
    public let variant: Variant
    public let precision: Precision

    /// Input-pixel ceiling for the single-pass (no tiles) path. Above it the frame is tiled.
    public let wholeFrameMaxPixels: Int

    /// Cooperative cancellation seam (CAN): called after every residual group of every forward — six times per
    /// tile / whole frame — and once per tile by the shared driver. Set by the package to `Task.checkCancellation`;
    /// a `CancellationError` thrown here reaches the caller unchanged.
    public var checkpoint: (() throws -> Void)?
    /// Progress seam (`RunProgress`): `(tilesDone, tilesTotal, groupsDone, groupsTotal)`.
    public var onProgress: ((Int, Int, Int, Int) -> Void)?

    // MARK: - Internals

    public let model: HEART
    private let tileProcessor: MLXTileProcessor

    // MARK: - Init

    /// Loads the variant's converted checkpoint from `weightsDirectory` (the flat store layout: one file per
    /// variant and lane), strictly, on the CPU stream. `load()` is residency, so the load is eager.
    public init(variant: Variant = .fidelity, precision: Precision = .fp16, weightsDirectory: URL,
                wholeFrameMaxPixels: Int = HEART_Playback.defaultWholeFrameMaxPixels,
                inputTileSize: Int = HEART_Playback.defaultInputTileSize,
                tileOverlap: Int = HEART_Playback.defaultTileOverlap,
                attentionHeadPad: Int = 64, residualStreamFloat32: Bool = false) throws {
        self.variant = variant
        self.precision = precision
        self.name = variant.tierName
        self.scaleFactor = variant.scale
        self.wholeFrameMaxPixels = wholeFrameMaxPixels
        self.inputTileSize = inputTileSize
        self.tileOverlap = tileOverlap

        var cfg = HEARTConfig(scale: variant.scale)
        cfg.attentionHeadPad = attentionHeadPad
        cfg.residualStreamFloat32 = residualStreamFloat32
        let url = weightsDirectory.appendingPathComponent(variant.fileName(precision: precision))
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw PlaybackTierError.weightsNotFound(url.path)
        }
        let model = HEART(config: cfg)
        do {
            try model.loadWeights(from: url)
        } catch {
            throw PlaybackTierError.modelLoadFailed(String(describing: error))
        }
        self.model = model
        self.tileProcessor = MLXTileProcessor(tileSize: inputTileSize, overlap: tileOverlap, scale: variant.scale)
    }

    /// Geometry defaults — MEASURED (PORTING-SPEC H5, 2026-09-28, the 27 Forge bench cells at 512² input):
    ///   • whole-frame up to 512² of input — the validated envelope (the bench's whole-frame 512² is what won every
    ///     damaged cell; larger whole frames are untested for quality and cost 6.7 / 11.7 / 25.8 GB fp16 at
    ///     960×540 / 1280×720 / 1920×1080). Hosts with the memory may raise it: a 540p whole frame runs 3.5 s
    ///     against 16 s tiled, because the shared driver's CPU compositing dominates the tiled path at ×4.
    ///   • 512² tiles above it: at the bench size one tile IS the whole frame (100.6 dB, i.e. 8-bit identical),
    ///     and no smaller tile keeps every cell within the ≤ 0.1 SSIMULACRA2 rule (384²: +0.20 mean but two
    ///     cells lose up to 0.47; 256²: seven cells lose, worst −2.02). The tiled path bounds fp16 memory at
    ///     3.2 GB through 1920×1080 → 7680×4320.
    ///   • overlap 32: 64 measured identical (±0.1 dB). ⚠️ Overlap is absolute (it covers the blend band, not a
    ///     receptive field — i-LN makes every tile's output depend on the whole tile anyway).
    public static let defaultWholeFrameMaxPixels = 512 * 512
    public static let defaultInputTileSize = 512
    public static let defaultTileOverlap = 32

    // MARK: - PlaybackTier impl

    public func upscale(_ buffer: CVPixelBuffer) async throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let tilesTotal = wholeFrameMaxPixels > 0 && width * height <= wholeFrameMaxPixels
            ? 1 : tileCount(width: width, height: height)
        var tilesDone = 0
        do {
            return try tileProcessor.processAdaptive(buffer, wholeFrameMaxPixels: wholeFrameMaxPixels) { tile in
                let y = try model.forward(tile, checkpoint: checkpoint) { g, gTotal in
                    self.onProgress?(tilesDone, tilesTotal, g, gTotal)
                }
                MLX.eval(y)
                tilesDone += 1
                return y
            }
        } catch let err as PlaybackTierError {
            throw err
        } catch is CancellationError {
            // CAN-2: never launder a cancellation — the checkpoint's CancellationError must reach the engine unchanged.
            throw CancellationError()
        } catch {
            throw PlaybackTierError.inferenceError(String(describing: error))
        }
    }

    /// The driver's deduplicated tile grid size (same clamping rule as `MLXTileProcessor.process`).
    public func tileCount(width: Int, height: Int) -> Int {
        func axis(_ n: Int) -> Int {
            var origins: [Int] = []
            for t in stride(from: 0, to: n, by: max(inputTileSize - tileOverlap, 1)) {
                let o = min(t, max(0, n - inputTileSize))
                if origins.last != o { origins.append(o) }
            }
            return max(origins.count, 1)
        }
        return axis(width) * axis(height)
    }

    /// One whole-frame forward on an NHWC float tensor (no tiles) — the CLI gates and the H5 tiling study use it.
    public func forward(_ x: MLXArray) throws -> MLXArray {
        try model.forward(x, checkpoint: checkpoint)
    }
}
