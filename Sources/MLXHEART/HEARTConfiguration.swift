import Foundation
import MLXToolKit
import HEARTMLX

/// Which released HEART checkpoint to load (Phips/HEART @ 868878ce4c25, re-hosted per lane on mlx-community).
public enum HEARTVariant: String, Codable, Sendable, CaseIterable {
    /// `heart_4x_otf_v2` — 4× OTF fidelity: the best damaged-input arm on the Forge bench, both oracles agreeing
    /// (NERVE-EVAL §5.7, AB-R-0370). Default.
    case fidelity
    /// `heart_4x_otf_gan` — 4× OTF GAN: RealPLKSR's texture level (+3.5 FR over it on damaged photo).
    case sharp
    /// `heart_4x_pretrain` — 4× official pretrain: clean sources.
    case clean
    /// `heart_2x` — 2× official pretrain: the only native ×2 checkpoint.
    case clean2x

    var coreVariant: HEART_Playback.Variant {
        switch self {
        case .fidelity: return .fidelity
        case .sharp: return .sharp
        case .clean: return .clean
        case .clean2x: return .clean2x
        }
    }
    /// Native scale of the checkpoint.
    public var scale: Int { coreVariant.scale }
}

/// Init-time configuration for `HEARTUpscalePackage` (C9). Stable for the session.
///
/// HEART ships ONE converted file per (variant, precision lane) — `mlx-community/HEART-fp16` (shipping lane, 33.7 MB
/// per variant) and `mlx-community/HEART-fp32` (parity lane, 66.8 MB) — so `quant` selects the lane and `variant`
/// the file: a configuration downloads only its own checkpoint (`WeightSourcing`, one role per variant × lane).
///
///   quant   file                              resident (weights)   role
///   .fp16   <ckpt>_fp16.safetensors           ≈ 34 MB              shipping lane — every reduction (i-LN statistics,
///                                                                  RIB tables, softmax, global pool + its head) in
///                                                                  fp32 inside; 67.6 dB vs the fp32 oracle, FR
///                                                                  indistinguishable from fp32 on the bench (Δ −0.01),
///                                                                  1.25× faster, 0.55× the memory (PORTING-SPEC H3)
///   .fp32   <ckpt>_fp32.safetensors           ≈ 67 MB              parity / reference lane (TF32 on M5 unless the
///                                                                  HOST sets MLX_ENABLE_TF32=0 — AB-L-0175: 63.6 dB
///                                                                  with the default, 126 dB with the flag)
///
/// bf16 is not offered (small-net mantissa posture: the fleet measured fp16 beating bf16 by 20 dB on FFTformer and
/// VOSR2 rejected bf16 at 37.6 dB); it resolves to fp16.
public struct HEARTConfiguration: PackageConfiguration, ModelStorable, QuantConfigured, BudgetAware {
    public var variant: HEARTVariant
    /// `.fp16` (default) or `.fp32`; anything else is treated as `.fp16`.
    public var quant: Quant

    /// Whole-frame fast-path ceiling in INPUT pixels, forwarded to the core; `nil` keeps the core default
    /// (`HEART_Playback.defaultWholeFrameMaxPixels` = 512², the bench-validated envelope). ⚠️ i-LN makes tiled
    /// output differ from whole-frame output (each tile is normalised by its own statistics), so this is a QUALITY
    /// knob as well as a memory knob — PORTING-SPEC H5 measured the defaults. Whole-frame memory (fp16, MLX peak):
    /// 3.5 GB at 512², 6.7 GB at 960×540, 11.7 GB at 1280×720, 25.8 GB at 1920×1080 (fp32 ≈ 2×); the tiled path
    /// stays under 3.2 GB at any size. Raising it trades memory for speed (540p: 3.5 s whole-frame vs 16 s tiled).
    public var wholeFrameMaxPixels: Int?
    /// Tile geometry for the tiled path; `nil` keeps the core defaults (H5-measured). Multiples of 32 recommended.
    public var inputTileSize: Int?
    public var tileOverlap: Int?

    /// Absolute path to a directory holding the converted files (`oracle/convert_weights.py` layout). **Honored
    /// OVER the engine-stamped store root** — dev-mode escape hatch, never touches the network.
    public var weightsDirectory: URL?
    /// Where the engine materializes weights — stamped from its `ModelStore.root` (`ModelStorable`).
    public var modelsRootDirectory: URL?
    /// Real headroom at load time, stamped by the governor (`BudgetAware`): below `fp32MinBudgetBytes` an fp32
    /// configuration loads the fp16 lane instead (same network, half the activation footprint).
    public var availableBudgetBytes: UInt64?

    /// The Hub repos a fresh machine fetches from — mlx-community's `<UpstreamRepoName>-<quant>` grammar, one repo
    /// per precision lane, every variant's file inside.
    public static let fp16Repo = "mlx-community/HEART-fp16"
    public static let fp32Repo = "mlx-community/HEART-fp32"
    public static func repo(for quant: Quant) -> String { quant == .fp32 ? fp32Repo : fp16Repo }
    /// Below this budget an fp32 request drops to fp16: the fp32 lane's declared working set is ~13.7 GB
    /// (1.09 GB resident + 12.6 GB activation peak on the in-app basis) against fp16's ~8.7 GB — see the manifest.
    public static let fp32MinBudgetBytes: UInt64 = 14_000_000_000

    public init(variant: HEARTVariant = .fidelity,
                quant: Quant = .fp16,
                wholeFrameMaxPixels: Int? = nil,
                inputTileSize: Int? = nil,
                tileOverlap: Int? = nil,
                weightsDirectory: URL? = nil,
                modelsRootDirectory: URL? = nil,
                availableBudgetBytes: UInt64? = nil) {
        self.variant = variant
        self.quant = quant
        self.wholeFrameMaxPixels = wholeFrameMaxPixels
        self.inputTileSize = inputTileSize
        self.tileOverlap = tileOverlap
        self.weightsDirectory = weightsDirectory
        self.modelsRootDirectory = modelsRootDirectory
        self.availableBudgetBytes = availableBudgetBytes
    }

    /// The precision lane actually loaded: `.fp32` as declared, everything else `.fp16`.
    public var effectiveQuant: Quant { quant == .fp32 ? .fp32 : .fp16 }
    public var precision: HEART_Playback.Precision { effectiveQuant == .fp32 ? .fp32 : .fp16 }

    /// The files a (variant, lane) needs.
    public static func files(for variant: HEARTVariant, quant: Quant) -> [String] {
        let lane: HEART_Playback.Precision = quant == .fp32 ? .fp32 : .fp16
        return [variant.coreVariant.fileName(precision: lane), "config.json"]
    }

    /// The repo this configuration's lane materializes from.
    public var repo: String { Self.repo(for: effectiveQuant) }

    /// The directory `load()` reads: the explicit `weightsDirectory`, else the store's flat repo directory
    /// (`<root>/models--mlx-community--HEART-fp16/`, where the engine's materializer lands files).
    public func resolvedWeightsDirectory(storeRoot: URL?) -> URL? {
        weightsDirectory ?? ModelStore(root: storeRoot).directory(for: repo)
    }

    // Persist only the portable knobs; stamped roots, budgets and absolute paths are per-session.
    private enum CodingKeys: String, CodingKey {
        case variant, quant, wholeFrameMaxPixels, inputTileSize, tileOverlap
    }
}

/// Fresh-machine sources (contract 1.24: the ENGINE downloads them into the store before `load()`). One role per
/// variant × lane, matching only that variant's file, so `.fidelity` never pulls the other three checkpoints.
extension HEARTConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        let lane = effectiveQuant == .fp32 ? "fp32" : "fp16"
        return [WeightSource(role: "heart-\(variant.rawValue)-\(lane)", repo: repo, revision: "main",
                             matching: Self.files(for: variant, quant: effectiveQuant))]
    }

    /// Explicit `weightsDirectory` first (complete → nothing missing, incomplete → the source is missing), then
    /// the MS-2 default probe over the store.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let dir = weightsDirectory {
            let complete = Self.files(for: variant, quant: effectiveQuant).allSatisfy {
                FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path)
            }
            return complete ? [] : weightSources
        }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }
}

/// Cold-start page-in of the variant's file from wherever it resolves.
extension HEARTConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        guard let dir = resolvedWeightsDirectory(storeRoot: modelsRootDirectory) else { return [] }
        return Self.files(for: variant, quant: effectiveQuant).map { dir.appendingPathComponent($0) }
    }
}
