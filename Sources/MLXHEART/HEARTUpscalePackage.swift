import Foundation
import Accelerate
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import RealESRGANMLX
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
/// **16 bits end to end** (contract 1.52.0, AB-A-0158): the surface declares `.rawRGBA16`. A `.rawRGBA16` image, or a
/// PNG/TIFF deeper than 8 bits, runs through the shared tile driver's 16-bit path and comes back at 16 bits:
/// `.rawRGBA16` for raw, a 16-bit PNG otherwise. 8-bit inputs take the 8-bit path byte for byte. The 16-bit path is
/// the same tiling as the 8-bit one (whole frame up to `wholeFrameMaxPixels`, then 512² tiles with a 32 px feathered
/// overlap, no halo): only the lift into the network and the write back change depth.
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
                // rounding error next to the process floor); the working set is the activations: 180-channel feature
                // maps through 36 blocks at the INPUT resolution plus the padded (q ‖ pos ‖ 0) Q/K/V of the window
                // attention and the 4× output buffer — bounded by the 512² tile geometry above `wholeFrameMaxPixels`.
                // Measured with `heart-smoke engine` through the REAL MLXServeEngine (release, M5 Max, MLX peak / floor,
                // one process per row, PORTING-SPEC H6, 2026-09-28):
                //
                //   input        output       path              lane   run       MLX peak   floor
                //   256²         1024²        whole-frame       fp16    0.9 s     1630 MB    61 MB
                //   512²         2048²        whole-frame       fp16    4.1 s*    2860 MB    61 MB
                //   960×540      3840×2160    tiled 512/32      fp16   16.0 s     3183 MB    61 MB
                //   1024²        4096²        tiled 512/32      fp16   35.5 s     3183 MB    61 MB
                //   1920×1080    7680×4320    tiled 512/32      fp16   46.9 s     3183 MB    61 MB
                //   512²         2048²        whole-frame       fp32    5.2 s     5298 MB   114 MB
                //   1920×1080    7680×4320    tiled 512/32      fp32   59.0 s     5167 MB   114 MB
                //   (* engine run incl. PNG codec; the model alone is 1.8 s. Whole-frame above the ceiling, if a host
                //    raises it: fp16 6.7 / 11.7 / 25.8 GB MLX peak at 960×540 / 1280×720 / 1920×1080, fp32 ≈ 2×.)
                //
                // DECLARED at the default envelope (512² tiles above 512² whole), converted from MLX-peak to the in-app
                // `phys_footprint` basis the governor compares against with the ratios the tile-driver SIBLING measured
                // on both bases (mlx-realesrgan-swift EFFICIENCY-ADOPTION.md: ~1.02 GB process floor; activation ×2.38):
                //   fp16: residentBytes ≈ 1.02 GB + 34 MB = 1.06 GB   peakActivationBytes ≈ 3.18 GB × 2.38 = 7.6 GB
                //   fp32: residentBytes ≈ 1.02 GB + 67 MB = 1.09 GB   peakActivationBytes ≈ 5.30 GB × 2.38 = 12.6 GB
                // ⚠️ PROVISIONAL — derived, not measured in-app. RE-BASELINE to real `phys_footprint` with the
                // image-fleet batch (AB-T-0019) and replace both numbers.
                footprints: [
                    QuantFootprint(quant: .fp16, residentBytes: 1_060_000_000, peakActivationBytes: 7_600_000_000),
                    QuantFootprint(quant: .fp32, residentBytes: 1_090_000_000, peakActivationBytes: 12_600_000_000),
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
                // Mirrors rawBGRA8 (since 0.1.0) and keeps 16 bits for rawRGBA16 and deep PNG (0.2.0).
                .declaring(rawImageFormats: [.rawBGRA8, .rawRGBA16])
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
        // Output mirrors the input format: rawBGRA8 in ⇒ rawBGRA8 out, rawRGBA16 in ⇒ rawRGBA16 out (no re-encode);
        // else .png — 16-bit when the input was deep, because the buffer then is.
        let outImage: Image
        if req.image.format == .rawBGRA8 {
            guard let raw = Self.encodeRawBGRA8(outPB) else { throw HEARTPackageError.imageEncodeFailed }
            outImage = raw
        } else if req.image.format == .rawRGBA16 {
            guard let raw = Self.encodeRawRGBA16(outPB) else { throw HEARTPackageError.imageEncodeFailed }
            outImage = raw
        } else {
            guard let png = Self.encodePNG(outPB) else { throw HEARTPackageError.imageEncodeFailed }
            outImage = Image(format: .png, data: png, width: w, height: h)
        }
        return ImageUpscaleResponse(image: outImage, appliedScale: appliedScale)
    }

    // MARK: - Image codec (identical to the Real-ESRGAN / RealPLKSR siblings' — the canonical Image ↔ BGRA seam)

    /// Decode a canonical `Image` (.png/.jpeg/.rawBGRA8/.rawRGBA16) to a `CVPixelBuffer`: BGRA, or the tile driver's
    /// 16-bit `deepPixelFormat` for `.rawRGBA16` and for an encoded image deeper than 8 bits per component.
    public nonisolated static func decodeToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        if image.format == .rawBGRA8 { return try rawBGRA8ToPixelBuffer(image) }
        if image.format == .rawRGBA16 { return try rawRGBA16ToPixelBuffer(image) }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw HEARTPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
        if cg.bitsPerComponent > 8 { return try deepPixelBuffer(drawing: cg) }
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

    /// Encode a BGRA `CVPixelBuffer` as PNG bytes (a 16-bit PNG for a `deepPixelFormat` buffer).
    public nonisolated static func encodePNG(_ pb: CVPixelBuffer) -> Data? {
        if MLXTileProcessor.isDeep(pb) {
            guard let cg = deepCGImage(pb) else { return nil }
            let out = NSMutableData()
            guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
            CGImageDestinationAddImage(dest, cg, nil)
            return CGImageDestinationFinalize(dest) ? out as Data : nil
        }
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

    /// Lanczos downsample of a `CVPixelBuffer` to `w`×`h`: a new buffer in the source's format, 32BGRA or the 16-bit
    /// `deepPixelFormat` (vImage, AB-T-0225). The CoreGraphics `.high` draw this replaces sampled on a corner-aligned
    /// grid, so a ×2 reduction drifted ±0.26 px from the centre to the edges. That cost 2.8 dB mean PSNR at ×2 (up to
    /// 6.3 dB on line art) against a Lanczos reduction of the same ×4 frame (AB-R-0486). vImage's Lanczos matches PIL's
    /// to 0.02 dB.
    public nonisolated static func resizePixelBuffer(_ src: CVPixelBuffer, toWidth w: Int, height h: Int) throws -> CVPixelBuffer {
        guard w > 0, h > 0 else { throw HEARTPackageError.imageEncodeFailed }
        let deep = MLXTileProcessor.isDeep(src)
        let buffer: CVPixelBuffer
        if deep {
            buffer = try makeDeepPixelBuffer(w, h)
        } else {
            var pb: CVPixelBuffer?
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: w,
                kCVPixelBufferHeightKey as String: h,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            ]
            guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
                  let made = pb else {
                throw HEARTPackageError.imageEncodeFailed
            }
            buffer = made
        }
        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }
        guard let srcBase = CVPixelBufferGetBaseAddress(src), let dstBase = CVPixelBufferGetBaseAddress(buffer) else {
            throw HEARTPackageError.imageEncodeFailed
        }
        var s = vImage_Buffer(data: srcBase, height: vImagePixelCount(CVPixelBufferGetHeight(src)),
                              width: vImagePixelCount(CVPixelBufferGetWidth(src)), rowBytes: CVPixelBufferGetBytesPerRow(src))
        var d = vImage_Buffer(data: dstBase, height: vImagePixelCount(h), width: vImagePixelCount(w),
                              rowBytes: CVPixelBufferGetBytesPerRow(buffer))
        // Four interleaved channels either way, each filtered alike, so BGRA and RGBA order both hold; premultiplied
        // alpha resamples correctly as is.
        let err = deep ? vImageScale_ARGB16U(&s, &d, nil, vImage_Flags(kvImageNoFlags))
                       : vImageScale_ARGB8888(&s, &d, nil, vImage_Flags(kvImageNoFlags))
        guard err == kvImageNoError else { throw HEARTPackageError.imageEncodeFailed }
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

    // MARK: - 16 bits (contract 1.52.0, AB-A-0158; the mlx-realplksr-swift 0.2.0 codec)

    /// RGBA, 16 bits per component, little-endian, premultiplied — `.rawRGBA16`'s layout as CoreGraphics draws it.
    public nonisolated static let rgba16BitmapInfo =
        CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder16Little.rawValue

    public nonisolated static func makeDeepPixelBuffer(_ w: Int, _ h: Int) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: MLXTileProcessor.deepPixelFormat,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, MLXTileProcessor.deepPixelFormat, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw HEARTPackageError.imageDecodeFailed("16-bit pixel buffer allocation (\(w)x\(h))")
        }
        return buffer
    }

    /// Draw `cg` into a new `deepPixelFormat` buffer, in sRGB at 16 bits (its own size unless given).
    public nonisolated static func deepPixelBuffer(drawing cg: CGImage, width: Int? = nil, height: Int? = nil,
                                            interpolation: CGInterpolationQuality = .default) throws -> CVPixelBuffer {
        let w = width ?? cg.width, h = height ?? cg.height
        let buffer = try makeDeepPixelBuffer(w, h)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let ctx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 16,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: rgba16BitmapInfo) else {
            throw HEARTPackageError.imageDecodeFailed("CGContext for RGBA16 draw")
        }
        ctx.interpolationQuality = interpolation
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Wrap raw interleaved RGBA16 bytes straight into a `deepPixelFormat` buffer — no decode.
    public nonisolated static func rawRGBA16ToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        guard let w = image.width, let h = image.height, w > 0, h > 0 else {
            throw HEARTPackageError.imageDecodeFailed("rawRGBA16 requires width/height")
        }
        let srcStride = image.bytesPerRow ?? (w * 8)
        guard srcStride >= w * 8, image.data.count >= srcStride * h else {
            throw HEARTPackageError.imageDecodeFailed(
                "rawRGBA16 data too small (\(image.data.count) < \(srcStride * h))")
        }
        let buffer = try makeDeepPixelBuffer(w, h)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw HEARTPackageError.imageDecodeFailed("pixel buffer base address")
        }
        let dstStride = CVPixelBufferGetBytesPerRow(buffer)
        image.data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let srcBase = src.baseAddress else { return }
            for row in 0..<h {
                memcpy(base.advanced(by: row * dstStride), srcBase.advanced(by: row * srcStride), w * 8)
            }
        }
        return buffer
    }

    /// A `deepPixelFormat` buffer as a 16-bit sRGB `CGImage` over a copy of its pixels.
    public nonisolated static func deepCGImage(_ pb: CVPixelBuffer) -> CGImage? {
        guard let raw = encodeRawRGBA16(pb), let provider = CGDataProvider(data: raw.data as CFData) else { return nil }
        return CGImage(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb), bitsPerComponent: 16,
                       bitsPerPixel: 64, bytesPerRow: CVPixelBufferGetWidth(pb) * 8,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: rgba16BitmapInfo),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }

    /// Emit a `deepPixelFormat` buffer as tightly-packed `.rawRGBA16` bytes.
    public nonisolated static func encodeRawRGBA16(_ pb: CVPixelBuffer) -> Image? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard w > 0, h > 0, MLXTileProcessor.isDeep(pb) else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let srcStride = CVPixelBufferGetBytesPerRow(pb)
        let dstStride = w * 8
        var out = Data(count: dstStride * h)
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            guard let dstBase = dst.baseAddress else { return }
            for row in 0..<h {
                memcpy(dstBase.advanced(by: row * dstStride), base.advanced(by: row * srcStride), dstStride)
            }
        }
        return Image.rawRGBA16(data: out, width: w, height: h)
    }
}

extension HEARTUpscalePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(HEARTUpscalePackage.self)
    }
}
