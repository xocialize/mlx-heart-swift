import Testing
import Foundation
import CoreGraphics
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLXToolKit
@testable import MLXHEART

/// Offline conformance checks — no Metal evaluation. Live upscaling is proven by `heart-smoke engine` (the real
/// engine path) and the gates in PORTING-SPEC.md.
struct HEARTUpscaleTests {

    @Test func manifestIsImageUpscaleAndPermissiveOnBothLayers() {
        let m = HEARTUpscalePackage.manifest
        #expect(m.capabilities == [.imageUpscale])
        #expect(m.license.weightLicense == .apache2)      // C7 — Phips/HEART, Apache-2.0, CC0-trained
        #expect(m.license.portCodeLicense == .apache2)    // C8 — isomorphic derivative of the Apache-2.0 arch files
        #expect(LicensePolicy.permissiveOnly.evaluate(m.license) == .admitted)
    }

    @Test func provenancePointsAtANamespaceWeControl() {
        // Fleet policy (2026-08-03): every weight source names a namespace WE control — mlx-community (member
        // org, the `<UpstreamRepoName>-<quant>` grammar) or xocialize.
        let ok: (String) -> Bool = { $0.hasPrefix("mlx-community/") || $0.hasPrefix("xocialize/") }
        #expect(ok(HEARTUpscalePackage.manifest.provenance.sourceRepo))
        #expect(ok(HEARTConfiguration.fp16Repo) && ok(HEARTConfiguration.fp32Repo))
        #expect(HEARTConfiguration(quant: .fp16).repo == HEARTConfiguration.fp16Repo)
        #expect(HEARTConfiguration(quant: .fp32).repo == HEARTConfiguration.fp32Repo)
        #expect(HEARTConfiguration(quant: .bf16).repo == HEARTConfiguration.fp16Repo)   // bf16 resolves to fp16
    }

    @Test func manifestRequirements() {
        let r = HEARTUpscalePackage.manifest.requirements
        #expect(r.requiredBackends.contains(.metalGPU))
        #expect(r.os.minMacOS == SemanticVersion(major: 26, minor: 0, patch: 0))
    }

    /// Efficiency adoption (engine 1.14): both lanes declare the split (weights floor + the activation peak), and
    /// the fp16 lane is the lighter of the two.
    @Test func splitFootprintsDeclaredPerLane() {
        let fps = HEARTUpscalePackage.manifest.requirements.footprints
        for q in [Quant.fp16, .fp32] {
            let fp = fps.first { $0.quant == q }
            #expect(fp != nil, "\(q) footprint missing")
            #expect((fp?.peakActivationBytes ?? 0) > 0 && (fp?.residentBytes ?? 0) > 0)
        }
        let fp16 = fps.first { $0.quant == .fp16 }!, fp32 = fps.first { $0.quant == .fp32 }!
        #expect(fp16.peakActivationBytes < fp32.peakActivationBytes)
    }

    @Test func quantConfiguredAndBudgetAware() {
        let cfg: any PackageConfiguration = HEARTConfiguration()
        #expect((cfg as? QuantConfigured)?.quant == .fp16)
        #expect(cfg is BudgetAware && cfg is ModelStorable && cfg is WeightSourcing && cfg is WeightPrewarming)
        #expect(HEARTConfiguration(quant: .bf16).effectiveQuant == .fp16)
        #expect(HEARTConfiguration(quant: .int8).effectiveQuant == .fp16)
        // Under a tight budget the fp32 plan drops to fp16; fp16 plans and roomy fp32 plans are untouched.
        #expect(HEARTUpscalePackage(configuration: HEARTConfiguration(quant: .fp32, availableBudgetBytes: 4_000_000_000)).plannedQuant == .fp16)
        #expect(HEARTUpscalePackage(configuration: HEARTConfiguration(quant: .fp32, availableBudgetBytes: 16_000_000_000)).plannedQuant == .fp32)
        #expect(HEARTUpscalePackage(configuration: HEARTConfiguration(quant: .fp16, availableBudgetBytes: 1_000_000_000)).plannedQuant == .fp16)
        #expect(HEARTUpscalePackage(configuration: HEARTConfiguration(quant: .fp32)).plannedQuant == .fp32)
    }

    @Test func surfaceIsTheCanonicalUpscaleDescriptor() {
        let s = HEARTUpscalePackage.manifest.surfaces.first
        #expect(s?.capability == .imageUpscale)
        #expect(s?.name == "heart-upscale")
        #expect(s?.parameters.first?.kind == .image)
        #expect(s?.parameters.contains { $0.name == "scale" && !$0.required } == true)
    }

    @Test func registrationConstructs() throws {
        let reg = HEARTUpscalePackage.registration
        #expect(reg.manifest.capabilities == [.imageUpscale])
        let pkg = try reg.makePackage(HEARTConfiguration())
        #expect(pkg is HEARTUpscalePackage)
    }

    /// One role per variant × lane, matching ONLY that variant's file (+ config.json).
    @Test func weightSourcesFollowTheVariantAndLane() {
        for v in HEARTVariant.allCases {
            let half = HEARTConfiguration(variant: v, quant: .fp16).weightSources
            #expect(half.count == 1)
            #expect(half[0].role == "heart-\(v.rawValue)-fp16" && half[0].repo == HEARTConfiguration.fp16Repo)
            #expect(half[0].matching == [v.coreVariant.fileName(precision: .fp16), "config.json"])
            let full = HEARTConfiguration(variant: v, quant: .fp32).weightSources
            #expect(full[0].role == "heart-\(v.rawValue)-fp32" && full[0].repo == HEARTConfiguration.fp32Repo)
            #expect(full[0].matching?.contains(v.coreVariant.fileName(precision: .fp32)) == true)
        }
        let roles = Set(HEARTVariant.allCases.flatMap { v in [Quant.fp16, .fp32].map { HEARTConfiguration(variant: v, quant: $0).weightSources[0].role } })
        #expect(roles.count == 8)
        #expect(HEARTVariant.clean2x.scale == 2 && HEARTVariant.fidelity.scale == 4)
    }

    @Test func configurationCodableRoundTrips() throws {
        let c = HEARTConfiguration(variant: .sharp, quant: .fp32, wholeFrameMaxPixels: 123, inputTileSize: 384,
                                   weightsDirectory: URL(fileURLWithPath: "/x"), modelsRootDirectory: URL(fileURLWithPath: "/y"),
                                   availableBudgetBytes: 1)
        let back = try JSONDecoder().decode(HEARTConfiguration.self, from: JSONEncoder().encode(c))
        #expect(back.variant == .sharp && back.quant == .fp32)
        #expect(back.wholeFrameMaxPixels == 123 && back.inputTileSize == 384)
        #expect(back.weightsDirectory == nil && back.modelsRootDirectory == nil && back.availableBudgetBytes == nil)
    }

    @Test func pngRoundTripsThroughPixelBuffer() throws {
        let png = try #require(Self.makePNG(width: 32, height: 32))
        let image = Image(format: .png, data: png, width: 32, height: 32)
        let pb = try HEARTUpscalePackage.decodeToPixelBuffer(image)
        #expect(CVPixelBufferGetWidth(pb) == 32)
        let back = try #require(HEARTUpscalePackage.encodePNG(pb))
        #expect(back.prefix(4) == Data([0x89, 0x50, 0x4E, 0x47]))
    }

    @Test func rawBGRA8RoundTripsBitIdentical() throws {
        let w = 8, h = 4
        let bytes = Data((0..<(w * h * 4)).map { UInt8($0 % 256) })
        let image = Image.rawBGRA8(data: bytes, width: w, height: h)
        let pb = try HEARTUpscalePackage.decodeToPixelBuffer(image)
        #expect(CVPixelBufferGetWidth(pb) == w && CVPixelBufferGetHeight(pb) == h)
        let back = try #require(HEARTUpscalePackage.encodeRawBGRA8(pb))
        #expect(back.format == .rawBGRA8)
        #expect(back.width == w && back.height == h && back.bytesPerRow == nil)
        #expect(back.data == bytes)
    }

    @Test func rawBGRA8MissingDimensionsThrows() {
        let image = Image(format: .rawBGRA8, data: Data(count: 16))
        #expect(throws: HEARTPackageError.self) {
            _ = try HEARTUpscalePackage.decodeToPixelBuffer(image)
        }
    }

    /// The sub-native `scale` path yields the requested dimensions as a valid 32BGRA buffer.
    @Test func resizePixelBufferProducesRequestedDimensions() throws {
        let png = try #require(Self.makePNG(width: 64, height: 64))
        let nativePB = try HEARTUpscalePackage.decodeToPixelBuffer(Image(format: .png, data: png, width: 64, height: 64))
        let scaled = try HEARTUpscalePackage.resizePixelBuffer(nativePB, toWidth: 32, height: 32)
        #expect(CVPixelBufferGetWidth(scaled) == 32 && CVPixelBufferGetHeight(scaled) == 32)
        #expect(CVPixelBufferGetPixelFormatType(scaled) == kCVPixelFormatType_32BGRA)
    }

    static func makePNG(width: Int, height: Int) -> Data? {
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.setFillColor(CGColor(gray: 0.6, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let cg = ctx.makeImage() else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }
}
