import Foundation
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import HEARTMLX

/// Errors at the HEART package boundary.
public enum HEARTPackageError: Error, Equatable {
    case imageDecodeFailed(String)
    case imageEncodeFailed
    case weightsDirectoryUnresolved
}

/// An MLXEngine `imageUpscale` package over **HEART** (Phips/HEART, Apache-2.0): Philip Hofmann's 16.7 M-parameter
/// HAT-iLN window-attention super-resolution network — the fidelity-class stills tier beside Real-ESRGAN (fast),
/// RealPLKSR (web-photo), VOSR2 (generative) and SeedVR2 (diffusion). Trained only on the CC0 LUCID corpus
/// (AB-D-0106); on the Forge bench its `.fidelity` checkpoint won every damaged cell against every arm
/// (NERVE-EVAL §5.7, AB-R-0370) at 4.5–5.1× RealPLKSR's cost.
///
/// A thin conformance wrapper over the standalone `HEARTMLX` core; all model logic (the network, i-LN, RIB window
/// attention, the shared tile driver, NHWC) lives there. Weights materialize from `mlx-community/HEART-{fp16,fp32}`
/// per variant (`WeightSourcing`); an explicit `weightsDirectory` is the dev-mode escape hatch.
///
/// Native scale is **4×** (`.fidelity` / `.sharp` / `.clean`) or **2×** (`.clean2x`). A request's `scale` of `nil`
/// or `≥ native` runs at native scale; a sub-native `scale` is honored by post-downsampling the native result to
/// `inputDim * scale`. The response's `appliedScale` always reports what actually ran.
///
/// Born sweep-clean (1.14): split footprint per lane, `QuantConfigured`, `BudgetAware` fp32 → fp16, `unload()`
/// flushes the MLX pool; born materialization-clean (`WeightSourcing`, engine-executed) and cancel-clean (entry
/// checkpoint + one per tile + one per residual GROUP — six per forward — with `RunProgress` at the same seams).
@InferenceActor
public final class HEARTUpscalePackage: ModelPackage {
    public typealias Configuration = HEARTConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: the weights are Apache-2.0 (Phips/HEART; trained only on Phips/lucid-cc0-v2-hc-512, CC0).
            // C8: the port is an isomorphic derivative of the upstream Apache-2.0 arch files → Apache-2.0 (NOTICE).
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(sourceRepo: HEARTConfiguration.fp16Repo, revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint (engine 1.14). resident = the loaded checkpoint (34 MB fp16 / 67 MB fp32 — a
                // rounding error); the working set is the activations: 180-channel feature maps through 36 blocks
                // at the INPUT resolution plus the padded (q ‖ pos ‖ 0) Q/K/V of the window attention and the 4×
                // output buffer, bounded by the tile geometry above `wholeFrameMaxPixels`.
                //
                // ⚠️ PROVISIONAL — placeholders until PORTING-SPEC H5/H6 measure `heart-smoke engine` at five sizes
                // (MLX-peak basis) and the image-fleet batch (AB-T-0019) re-baselines to in-app `phys_footprint`.
                footprints: [
                    QuantFootprint(quant: .fp16, residentBytes: 1_100_000_000, peakActivationBytes: 4_000_000_000),
                    QuantFootprint(quant: .fp32, residentBytes: 1_150_000_000, peakActivationBytes: 8_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                ImageUpscaleContract.descriptor(
                    name: "heart-upscale",
                    summary: "HEART 4x super-resolution (HAT-iLN window attention, 16.7M params, CC0-trained): the fidelity tier for damaged photos and screenshots — best detail recovery on blurred, noisy or JPEG'd sources at ~5x RealPLKSR's cost; native 2x via the clean2x variant."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var upscaler: HEART_Playback?
    private var loadedQuant: Quant?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    /// The lane `load()` will use: the configured one, unless it is fp32 and the governor-stamped budget is below
    /// `fp32MinBudgetBytes` — then fp16 (same network, half the activation footprint).
    public nonisolated var plannedQuant: Quant {
        if configuration.effectiveQuant == .fp32, let b = configuration.availableBudgetBytes,
           b < HEARTConfiguration.fp32MinBudgetBytes {
            return .fp16
        }
        return configuration.effectiveQuant
    }

    public func load() async throws {
        guard upscaler == nil else { return }
        guard let dir = configuration.resolvedWeightsDirectory(storeRoot: configuration.modelsRootDirectory) else {
            throw HEARTPackageError.weightsDirectoryUnresolved
        }
        let quant = plannedQuant
        // Contract 1.24: the engine has already materialized the variant's file into `dir`; this just loads
        // (strict key contract, CPU-stream load, RIB tables rebuilt from the loaded weights).
        upscaler = try HEART_Playback(
            variant: configuration.variant.coreVariant,
            precision: quant == .fp32 ? .fp32 : .fp16,
            weightsDirectory: dir,
            wholeFrameMaxPixels: configuration.wholeFrameMaxPixels ?? HEART_Playback.defaultWholeFrameMaxPixels,
            inputTileSize: configuration.inputTileSize ?? HEART_Playback.defaultInputTileSize,
            tileOverlap: configuration.tileOverlap ?? HEART_Playback.defaultTileOverlap)
        loadedQuant = quant
    }

    /// C14 seam: the loaded graph by role (`nil` before `load()`), for the INF gate's walker.
    var inferenceModeGraphs: [String: HEARTMLX.HEART?] { ["model": upscaler?.model] }

    public func unload() async {
        upscaler = nil
        loadedQuant = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS (not just drop refs)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation.
        try Task.checkCancellation()
        guard let upscaler else { throw PackageError.notLoaded }
        guard request.capability == .imageUpscale,
              let req = request as? ImageUpscaleRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        let inPB = try Self.decodeToPixelBuffer(req.image)
        let inW = CVPixelBufferGetWidth(inPB), inH = CVPixelBufferGetHeight(inPB)
        let native = upscaler.scaleFactor

        // CAN-2: one cooperative checkpoint per tile (the shared driver) and per residual group (the core),
        // rethrown unchanged; RunProgress at the same seams (1.18): step = tiles, stage = groups.
        upscaler.checkpoint = { try Task.checkCancellation() }
        upscaler.onProgress = { tile, tiles, group, groups in
            RunProgress.report(.upsample, step: tile * groups + group, totalSteps: tiles * groups,
                               stage: group, totalStages: groups)
        }
        defer { upscaler.checkpoint = nil; upscaler.onProgress = nil }

        let nativePB = try await upscaler.upscale(inPB)

        // Honor a requested `scale` below the native factor by post-downsampling the native result (BRIDGE-029).
        // `nil`, the native factor, or any request ≥ native pass through at native scale.
        let outPB: CVPixelBuffer
        let appliedScale: Int
        if let s = req.scale, s > 0, s < native {
            outPB = try Self.resizePixelBuffer(nativePB, toWidth: inW * s, height: inH * s)
            appliedScale = s
        } else {
            outPB = nativePB
            appliedScale = native
        }

        let w = CVPixelBufferGetWidth(outPB), h = CVPixelBufferGetHeight(outPB)
        // Output mirrors the input format: rawBGRA8 in ⇒ rawBGRA8 out (no re-encode); else .png.
        let outImage: Image
        if req.image.format == .rawBGRA8 {
            guard let raw = Self.encodeRawBGRA8(outPB) else { throw HEARTPackageError.imageEncodeFailed }
            outImage = raw
        } else {
            guard let png = Self.encodePNG(outPB) else { throw HEARTPackageError.imageEncodeFailed }
            outImage = Image(format: .png, data: png, width: w, height: h)
        }
        return ImageUpscaleResponse(image: outImage, appliedScale: appliedScale)
    }

    // MARK: - Image codec (identical to the Real-ESRGAN / RealPLKSR siblings' — the canonical Image ↔ BGRA seam)

    /// Decode a canonical `Image` (.png/.jpeg/.rawBGRA8) to a BGRA `CVPixelBuffer`.
    public nonisolated static func decodeToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        if image.format == .rawBGRA8 { return try rawBGRA8ToPixelBuffer(image) }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw HEARTPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
        let w = cg.width, h = cg.height
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw HEARTPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let ctx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw HEARTPackageError.imageDecodeFailed("CGContext for BGRA draw")
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Encode a BGRA `CVPixelBuffer` as PNG bytes.
    public nonisolated static func encodePNG(_ pb: CVPixelBuffer) -> Data? {
        let ci = CIImage(cvPixelBuffer: pb)
        let ctx = CIContext(options: [.cacheIntermediates: false])
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }

    /// Wrap raw interleaved BGRA8 bytes straight into a 32BGRA `CVPixelBuffer` — no decode.
    public nonisolated static func rawBGRA8ToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        guard let w = image.width, let h = image.height, w > 0, h > 0 else {
            throw HEARTPackageError.imageDecodeFailed("rawBGRA8 requires width/height")
        }
        let srcStride = image.bytesPerRow ?? (w * 4)
        guard srcStride >= w * 4, image.data.count >= srcStride * h else {
            throw HEARTPackageError.imageDecodeFailed(
                "rawBGRA8 data too small (\(image.data.count) < \(srcStride * h))")
        }
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw HEARTPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw HEARTPackageError.imageDecodeFailed("pixel buffer base address")
        }
        let dstStride = CVPixelBufferGetBytesPerRow(buffer)
        let rowBytes = min(srcStride, dstStride)
        image.data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let srcBase = src.baseAddress else { return }
            for row in 0..<h {
                memcpy(base.advanced(by: row * dstStride), srcBase.advanced(by: row * srcStride), rowBytes)
            }
        }
        return buffer
    }

    /// High-quality downsample of a 32BGRA `CVPixelBuffer` to `w`×`h` (a new 32BGRA buffer).
    public nonisolated static func resizePixelBuffer(_ src: CVPixelBuffer, toWidth w: Int, height h: Int) throws -> CVPixelBuffer {
        guard w > 0, h > 0 else { throw HEARTPackageError.imageEncodeFailed }
        let ci = CIImage(cvPixelBuffer: src)
        let ctx = CIContext(options: [.cacheIntermediates: false])
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { throw HEARTPackageError.imageEncodeFailed }
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw HEARTPackageError.imageEncodeFailed
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let outCtx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw HEARTPackageError.imageEncodeFailed
        }
        outCtx.interpolationQuality = .high
        outCtx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Emit a 32BGRA `CVPixelBuffer` as tightly-packed raw BGRA8 `Image` bytes.
    public nonisolated static func encodeRawBGRA8(_ pb: CVPixelBuffer) -> Image? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard w > 0, h > 0 else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let srcStride = CVPixelBufferGetBytesPerRow(pb)
        let dstStride = w * 4
        var out = Data(count: dstStride * h)
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            guard let dstBase = dst.baseAddress else { return }
            for row in 0..<h {
                memcpy(dstBase.advanced(by: row * dstStride), base.advanced(by: row * srcStride), dstStride)
            }
        }
        return Image.rawBGRA8(data: out, width: w, height: h)
    }
}

extension HEARTUpscalePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(HEARTUpscalePackage.self)
    }
}
