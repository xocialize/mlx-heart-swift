// heart-smoke — the CLI gate lane for the HEART port (PORTING-SPEC.md phases S0–S7) and the real-engine smoke.
//
//   heart-smoke keys <weightsDir> [fp32|fp16]                     S0: key contract per checkpoint, 0 missing / 0 unused
//   heart-smoke gate <weightsDir> <goldensDir> [--gpu] [--head 40|64] [--fp16] [--residual-fp32] [--only taps,e2e] [--sizes 128,512] [--variants a,b]
//                                                                  S1/S2: tables + per-sub-op parity vs the PyTorch
//                                                                  CPU-fp32 oracle, e2e vs torch and the author's ONNX
//   heart-smoke run <in.png> <out.png> <weightsDir> [--variant fidelity|sharp|clean|clean2x] [--fp32] [--cpu]
//                  [--head N] [--tile N] [--overlap N] [--whole N] [--residual-fp32]      S2b: a real image
//   heart-smoke engine <in.png> <out.png> <weightsDir> [--variant V] [--fp32] [--scale N] [--raw] [--whole N] [--tile N] [--cpu]
//                                                                  S7: HEARTUpscalePackage through the REAL
//                                                                  MLXServeEngine (register → run), explicit dir
//   heart-smoke perf <weightsDir> [--sizes 256,512] [--rounds 5]   H4: idle-GPU timing bracket, arms interleaved
//   heart-smoke bench <weightsDir> <benchDir> <outDir> [--arms fp32-64-whole,fp16-64-whole,…] [--json f]
//                                                                  H3/H5: dtype + tiling arms on the 27 bench cells
//   heart-smoke bench <weightsDir> --memory-curve                  H5: whole-frame peak memory at four input sizes
//   heart-smoke cancel <weightsDir> [--after 1.5]                  live mid-run cancel probe (type + latency)
//
// Gates run on the CPU stream by default (the numerically strict lane; fp32 goldens). `--gpu` reruns them on Metal —
// on M5 set MLX_ENABLE_TF32=0 in the environment first (AB-L-0175) or the fp32 lane reads TF32-class.

import Foundation
import CoreGraphics
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXNN
import MLXToolKit
import MLXServeCore
import HEARTMLX
import MLXHEART

enum SmokeError: Error, CustomStringConvertible {
    case usage(String), badImage(String), gateFailed([String]), missing(String)
    var description: String {
        switch self {
        case .usage(let s): return "usage: \(s)"
        case .badImage(let s): return "bad image: \(s)"
        case .gateFailed(let f): return "GATE FAILED: \(f.joined(separator: ", "))"
        case .missing(let s): return "missing: \(s)"
        }
    }
}

func log(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
func now() -> Double { Date().timeIntervalSince1970 }
func flag(_ args: [String], _ name: String) -> Bool { args.contains(name) }
func opt(_ args: [String], _ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

// MARK: - comparison

struct Delta: CustomStringConvertible {
    let name: String
    let maxAbs: Float
    let meanAbs: Float
    let refMax: Float
    let cosine: Float
    let shapeOK: Bool
    let nDiff: Int
    var relMax: Float { refMax > 0 ? maxAbs / refMax : maxAbs }
    var description: String {
        String(format: "%-26@ max|Δ| %.3e  mean|Δ| %.3e  |ref|max %.4g  rel %.2e  cos %.8f  n≠ %d%@",
               name as NSString, maxAbs, meanAbs, refMax, relMax, cosine, nDiff, shapeOK ? "" : "  SHAPE MISMATCH")
    }
}

func compare(_ name: String, _ got: MLXArray, _ ref: MLXArray) -> Delta {
    let g = got.asType(.float32), r = ref.asType(.float32)
    guard g.shape == r.shape else {
        log("  \(name): shape \(g.shape) vs golden \(r.shape)")
        return Delta(name: name, maxAbs: .infinity, meanAbs: .infinity, refMax: 0, cosine: 0, shapeOK: false, nDiff: -1)
    }
    let d = abs(g - r)
    let num = sum(g * r).item(Float.self)
    let den = (sqrt(sum(g * g)) * sqrt(sum(r * r))).item(Float.self)
    return Delta(name: name, maxAbs: d.max().item(Float.self), meanAbs: d.mean().item(Float.self),
                 refMax: abs(r).max().item(Float.self), cosine: den > 0 ? num / den : 0, shapeOK: true,
                 nDiff: sum(d .> 0).item(Int32.self).description.count > 0 ? Int(sum(d .> 0).item(Int32.self)) : 0)
}

/// PSNR in dB for signals in [0, 1] (peak 1), on the raw (unclamped) outputs — the definition of the forge's
/// `heart_parity.py`, so the numbers are comparable with its 86.6–91.2 dB torch-vs-ONNX figures.
func psnr(_ got: MLXArray, _ ref: MLXArray) -> Float {
    let mse = mean(square(got.asType(.float32) - ref.asType(.float32))).item(Float.self)
    return mse > 0 ? 10 * log10f(1.0 / mse) : .infinity
}

func withDevice<R>(gpu: Bool, _ body: () throws -> R) rethrows -> R {
    try Device.withDefaultDevice(gpu ? Device(.gpu) : Device(.cpu), body)
}

func gpuUtilization() -> Int {
    let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/sbin/ioreg")
    p.arguments = ["-r", "-d", "1", "-c", "AGXAccelerator"]
    let pipe = Pipe(); p.standardOutput = pipe
    do { try p.run() } catch { return -1 }
    let data = pipe.fileHandleForReading.readDataToEndOfFile(); p.waitUntilExit()
    let s = String(decoding: data, as: UTF8.self)
    guard let r = s.range(of: "\"Device Utilization %\"=") else { return -1 }
    let tail = s[r.upperBound...].prefix(while: { $0.isNumber })
    return Int(tail) ?? -1
}

// MARK: - image I/O (CLI only; the core is MLX-only)

func loadRGB(_ path: String) throws -> MLXArray {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw SmokeError.badImage(path) }
    let w = cg.width, h = cg.height
    var buf = [UInt8](repeating: 0, count: w * h * 4)
    guard let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { throw SmokeError.badImage(path) }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    var f = [Float](repeating: 0, count: w * h * 3)
    for i in 0 ..< (w * h) { for c in 0 ..< 3 { f[i * 3 + c] = Float(buf[i * 4 + c]) / 255 } }
    return MLXArray(f, [1, h, w, 3])
}

func savePNG(_ x: MLXArray, to path: String) throws {
    let y = clip(x[0].asType(.float32), min: 0, max: 1) * 255 + 0.5
    let h = y.dim(0), w = y.dim(1)
    let v = y.asArray(Float.self)
    var buf = [UInt8](repeating: 255, count: w * h * 4)
    for i in 0 ..< (w * h) { for c in 0 ..< 3 { buf[i * 4 + c] = UInt8(min(255, max(0, Int(v[i * 3 + c])))) } }
    guard let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
          let cg = ctx.makeImage() else { throw SmokeError.badImage(path) }
    let out = NSMutableData()
    guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { throw SmokeError.badImage(path) }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw SmokeError.badImage(path) }
    try (out as Data).write(to: URL(fileURLWithPath: path))
}

func loadPNGData(_ path: String) throws -> Image {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw SmokeError.badImage(path) }
    let out = NSMutableData()
    guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { throw SmokeError.badImage(path) }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw SmokeError.badImage(path) }
    return Image(format: .png, data: out as Data, width: cg.width, height: cg.height)
}

// MARK: - model construction

func variant(_ args: [String]) -> HEART_Playback.Variant {
    HEART_Playback.Variant(rawValue: opt(args, "--variant") ?? "fidelity") ?? .fidelity
}

func makeModel(weightsDir: String, variant v: HEART_Playback.Variant, precision: HEART_Playback.Precision,
               headPad: Int, residualFP32: Bool = false) throws -> HEART {
    var cfg = HEARTConfig(scale: v.scale)
    cfg.attentionHeadPad = headPad
    cfg.residualStreamFloat32 = residualFP32
    let m = HEART(config: cfg)
    let url = URL(fileURLWithPath: weightsDir).appendingPathComponent(v.fileName(precision: precision))
    let t0 = now()
    try m.loadWeights(from: url)
    log(String(format: "loaded %@ (%@, head %d) in %.2f s", v.fileName(precision: precision), precision.rawValue, headPad, now() - t0))
    return m
}

// MARK: - S0 keys

func runKeys(_ args: [String]) throws {
    guard args.count >= 1 else { throw SmokeError.usage("keys <weightsDir> [fp32|fp16]") }
    let dir = URL(fileURLWithPath: args[0])
    let lanes: [HEART_Playback.Precision] = args.count > 1 ? [HEART_Playback.Precision(rawValue: args[1])!] : [.fp32, .fp16]
    var failures: [String] = []
    for v in HEART_Playback.Variant.allCases {
        for lane in lanes {
            let url = dir.appendingPathComponent(v.fileName(precision: lane))
            guard FileManager.default.fileExists(atPath: url.path) else { failures.append("\(url.lastPathComponent) missing"); continue }
            let arrays = try MLX.loadArrays(url: url)
            let scale = HEART.scale(ofCheckpointKeys: arrays.keys)
            let expected = HEART.expectedKeys(scale: scale)
            let got = Set(arrays.keys)
            let missing = expected.subtracting(got).sorted(), extra = got.subtracting(expected).sorted()
            let params = arrays.values.reduce(0) { $0 + $1.size }
            var dtypes: [String: Int] = [:]
            for a in arrays.values { dtypes[String(describing: a.dtype), default: 0] += 1 }
            // shapes must also load: a strict update on a fresh module
            var shapeErr = ""
            do {
                let m = HEART(config: HEARTConfig(scale: scale))
                try m.loadWeights(arrays)
            } catch { shapeErr = " LOAD ERROR: \(error)" }
            let ok = missing.isEmpty && extra.isEmpty && shapeErr.isEmpty
                && arrays.count == (scale == 4 ? HEART.tensorCountX4 : HEART.tensorCountX2)
            print(String(format: "%@ %@ scale %d: %d tensors, %d params, missing %d, unused %d, dtypes %@%@",
                         ok ? "PASS" : "FAIL", url.lastPathComponent, scale, arrays.count, params, missing.count, extra.count,
                         dtypes.description, shapeErr))
            if !ok { failures.append(url.lastPathComponent + (missing.isEmpty ? "" : " missing \(missing.prefix(3))") + (extra.isEmpty ? "" : " extra \(extra.prefix(3))")) }
        }
    }
    if !failures.isEmpty { throw SmokeError.gateFailed(failures) }
}

// MARK: - S1/S2 gate

struct GateResult {
    var passed: [String] = []
    var failed: [String] = []
    mutating func check(_ d: Delta, rel tol: Float, note: String = "") {
        let ok = d.shapeOK && d.relMax <= tol && d.maxAbs.isFinite
        print("  \(ok ? "PASS" : "FAIL") \(d) tol \(String(format: "%.0e", tol))\(note)")
        if ok { passed.append(d.name) } else { failed.append(d.name) }
    }
    mutating func checkExact(_ d: Delta) {
        let ok = d.shapeOK && d.maxAbs == 0
        print("  \(ok ? "PASS" : "FAIL") \(d) tol 0 (exact)")
        if ok { passed.append(d.name) } else { failed.append(d.name) }
    }
    mutating func checkUlp(_ d: Delta, maxUlp: Float, label: String) {
        let ok = d.shapeOK && d.maxAbs <= maxUlp
        print("  \(ok ? "PASS" : "FAIL") \(d) ≤ 1 ulp (\(label))")
        if ok { passed.append(d.name) } else { failed.append(d.name) }
    }
}

/// The per-sub-op rungs on one taps file, driving the model's sub-modules directly (the module path for each
/// composite is gated too, so a bug in the composition is not hidden by the piecewise path).
func gateTaps(model m: HEART, goldens g: [String: MLXArray], size: String, r: inout GateResult) {
    let PRIM: Float = 2e-6, DEEP: Float = 1e-5, CHAIN: Float = 5e-5, E2E: Float = 2e-4
    print("── taps \(size) ──")
    let input = g["in"]!
    let x = checkImageSize(input, window: m.config.windowSize)
    r.checkExact(compare("padded", x, g["padded"]!))
    let (f, t) = m.stem(x)
    r.check(compare("conv_first", f, g["conv_first"]!), rel: PRIM)
    r.check(compare("patch_embed", t, g["patch_embed"]!), rel: PRIM)
    let blk0 = m.layers[0].residualGroup.blocks[0]
    let (xn, std1) = blk0.norm1(t)
    r.check(compare("g0b0.norm1.x", xn, g["g0b0.norm1.x"]!), rel: DEEP)
    r.check(compare("g0b0.norm1.std", std1.reshaped([-1]), g["g0b0.norm1.std"]!), rel: DEEP)
    r.check(compare("g0b0.cab_pre_ca", blk0.convBlock.preAttention(xn), g["g0b0.cab_pre_ca"]!), rel: DEEP)
    let cab = blk0.convBlock(xn)
    r.check(compare("g0b0.cab", cab, g["g0b0.cab"]!), rel: DEEP)
    let att = blk0.attn!(xn)
    r.check(compare("g0b0.attn", att, g["g0b0.attn"]!), rel: DEEP)
    var y = t + std1 * (cab * blk0.convScale + att)
    let (xn2, std2) = blk0.norm2(y)
    r.check(compare("g0b0.norm2.x", xn2, g["g0b0.norm2.x"]!), rel: DEEP)
    r.check(compare("g0b0.norm2.std", std2.reshaped([-1]), g["g0b0.norm2.std"]!), rel: DEEP)
    let mlp = blk0.mlp(xn2)
    r.check(compare("g0b0.mlp", mlp, g["g0b0.mlp"]!), rel: DEEP)
    y = y + std2 * mlp
    r.check(compare("g0b0.out(piecewise)", y, g["g0b0.out"]!), rel: DEEP)
    let y0 = blk0(t)
    r.check(compare("g0b0.out(module)", y0, g["g0b0.out"]!), rel: DEEP)
    let y1 = m.layers[0].residualGroup.blocks[1](y0)
    r.check(compare("g0b1.out", y1, g["g0b1.out"]!), rel: DEEP)
    let blk2 = m.layers[0].residualGroup.blocks[2]
    let (xn21, _) = blk2.norm1(y1)
    // the reflect-shift tensor at tolerance 0: re-pad the ORACLE's own attention input (the golden's centre crop),
    // so the check isolates the pad op from the fp32 rounding of the normalised input feeding it
    let sp = g["g0b2.shiftpad"]!, s16 = m.config.windowSize / 2
    let oracleIn = sp[0..., s16 ..< (sp.dim(1) - s16), s16 ..< (sp.dim(2) - s16), 0...]
    r.checkExact(compare("g0b2.shiftpad", blk2.attn!.shiftPadded(oracleIn), sp))
    r.check(compare("g0b2.shiftpad(ported in)", blk2.attn!.shiftPadded(xn21), sp), rel: DEEP)
    r.check(compare("g0b2.attn(shifted)", blk2.attn!(xn21), g["g0b2.attn"]!), rel: DEEP)
    let y2 = blk2(y1)
    r.check(compare("g0b2.out", y2, g["g0b2.out"]!), rel: DEEP)
    let g0 = m.layers[0](t)
    r.check(compare("g0.out (RHAG 0)", g0, g["g0.out"]!), rel: DEEP)
    // g5b4 attention: chain through the groups, tap the last group's last attention block
    var z = t
    for gi in 0 ..< 5 { z = m.layers[gi](z) }
    let z5in = z
    let blocks5 = m.layers[5].residualGroup.blocks
    for bi in 0 ..< 4 { z = blocks5[bi](z) }
    let (xn54, _) = blocks5[4].norm1(z)
    r.check(compare("g5b4.attn (chained)", blocks5[4].attn!(xn54), g["g5b4.attn"]!), rel: CHAIN)
    for bi in 4 ..< 6 { z = blocks5[bi](z) }
    z = m.layers[5].conv(z) + z5in                                   // RHAG 5 closes with conv + residual
    let feats = m.norm(z)
    r.check(compare("features (chained)", feats, g["features"]!), rel: CHAIN)
    let cab2 = m.convAfterBody(feats)
    r.check(compare("conv_after_body", cab2, g["conv_after_body"]!), rel: CHAIN)
    let res = cab2 + f
    r.check(compare("after_body_residual", res, g["after_body_residual"]!), rel: CHAIN)
    let bu = leakyRelu((m.convBeforeUpsample[0] as! Conv2d)(res), negativeSlope: 0.01)
    r.check(compare("before_upsample", bu, g["before_upsample"]!), rel: CHAIN)
    var up = bu
    for stage in 0 ..< m.config.upsampleStages { up = pixelShuffleNHWC((m.upsample[2 * stage] as! Conv2d)(up), 2) }
    r.check(compare("upsample", up, g["upsample"]!), rel: CHAIN)
    let cl = m.convLast(up)
    r.check(compare("conv_last", cl, g["conv_last"]!), rel: CHAIN)
    let out = m(input)
    let d = compare("out (module forward)", out, g["out"]!)
    r.check(d, rel: E2E, note: String(format: "  PSNR %.1f dB", psnr(out, g["out"]!)))
}

func gateTables(model m: HEART, goldens g: [String: MLXArray], r: inout GateResult) {
    print("── tables (tolerance 0 / 1 ulp) ──")
    let coords = RIBCoordinates.shared(window: m.config.windowSize, nFreqs: m.config.ribNFreqs).coords
    r.checkUlp(compare("rib_coords", coords, g["rib_coords"]!), maxUlp: 5.9604645e-08,
               label: "torch float32 sin/cos are 1-ulp, not correctly rounded")
    var gi = 0
    for (li, layer) in m.layers.enumerated() {
        for (bi, blk) in layer.residualGroup.blocks.enumerated() {
            guard let attn = blk.attn else { continue }
            let (q, k) = attn.positionFeatures()
            r.check(compare("g\(li)b\(bi).q_pos", q, g["g\(li)b\(bi).q_pos"]!), rel: 2e-6)
            r.check(compare("g\(li)b\(bi).k_pos", k, g["g\(li)b\(bi).k_pos"]!), rel: 2e-6)
            gi += 1
        }
    }
    // the head-pad probe: padding (q‖pos) 38 → 64 with zero columns must not change the attention (vs 40)
}

func runGate(_ args: [String]) throws {
    guard args.count >= 2 else { throw SmokeError.usage("gate <weightsDir> <goldensDir> [--gpu] [--head 40|64] [--fp16] [--only taps,e2e] [--sizes 128,512]") }
    let weightsDir = args[0], goldensDir = args[1]
    let gpu = flag(args, "--gpu")
    let headPad = Int(opt(args, "--head") ?? "64")!
    let lane: HEART_Playback.Precision = flag(args, "--fp16") ? .fp16 : .fp32
    let residualFP32 = flag(args, "--residual-fp32")
    let only = (opt(args, "--only") ?? "taps,e2e").split(separator: ",").map(String.init)
    let sizes = (opt(args, "--sizes") ?? "128,512").split(separator: ",").map { Int($0)! }
    print("gate: device \(gpu ? "GPU" : "CPU"), lane \(lane.rawValue)\(residualFP32 ? " (residual stream fp32)" : ""), head pad \(headPad), MLX_ENABLE_TF32=\(ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "unset")")
    var r = GateResult()
    try withDevice(gpu: gpu) {
        if only.contains("taps") {
            let m = try makeModel(weightsDir: weightsDir, variant: .fidelity, precision: lane, headPad: headPad)
            let files = ["64x64", "96x96", "100x140", "20x20", "12x12"]
            var first = true
            for size in files {
                let path = "\(goldensDir)/taps_heart_4x_otf_v2_\(size).safetensors"
                guard FileManager.default.fileExists(atPath: path) else { log("skip \(path)"); continue }
                let g = try MLX.loadArrays(url: URL(fileURLWithPath: path))
                if first { gateTables(model: m, goldens: g, r: &r); first = false }
                let t0 = now()
                gateTaps(model: m, goldens: g, size: size, r: &r)
                log(String(format: "  taps %@ in %.1f s", size, now() - t0))
            }
            // head-pad probe: 40 (upstream) vs the configured pad on the same input — identical math
            if let g = try? MLX.loadArrays(url: URL(fileURLWithPath: "\(goldensDir)/taps_heart_4x_otf_v2_64x64.safetensors")) {
                let m40 = try makeModel(weightsDir: weightsDir, variant: .fidelity, precision: lane, headPad: 40)
                let a = m(g["in"]!), b = m40(g["in"]!)
                let d = compare("headpad \(headPad) vs 40 (e2e 64²)", a, b)
                r.check(d, rel: 1e-5, note: String(format: "  PSNR %.1f dB — zero columns, identical math", psnr(a, b)))
            }
        }
        if only.contains("e2e") {
            print("── e2e vs torch (CPU fp32) and the author's ONNX (ORT CPU) ──")
            let variants = opt(args, "--variants").map { $0.split(separator: ",").map { HEART_Playback.Variant(rawValue: String($0))! } }
                ?? HEART_Playback.Variant.allCases
            for v in variants {
                let m = try makeModel(weightsDir: weightsDir, variant: v, precision: lane, headPad: headPad, residualFP32: residualFP32)
                for side in sizes {
                    let path = "\(goldensDir)/e2e_\(v.checkpointStem)_\(side)x\(side).safetensors"
                    guard FileManager.default.fileExists(atPath: path) else { log("skip \(path)"); continue }
                    let g = try MLX.loadArrays(url: URL(fileURLWithPath: path))
                    let t0 = now()
                    let out = m(g["in"]!); eval(out)
                    let dt = now() - t0
                    let dTorch = compare("\(v.rawValue) \(side)² vs torch", out, g["out"]!)
                    let pT = psnr(out, g["out"]!), pO = psnr(out, g["onnx"]!), pTO = psnr(g["out"]!, g["onnx"]!)
                    let ok = pT >= 90
                    print(String(format: "  %@ %@  PSNR vs torch %.1f dB · vs ONNX %.1f dB (torch-vs-ONNX %.1f dB) · %.1f s",
                                 ok ? "PASS" : "FAIL", dTorch.description, pT, pO, pTO, dt))
                    if ok { r.passed.append(dTorch.name) } else { r.failed.append(dTorch.name) }
                }
            }
        }
    }
    print("gate summary: \(r.passed.count) passed, \(r.failed.count) failed")
    if !r.failed.isEmpty { throw SmokeError.gateFailed(r.failed) }
}

// MARK: - run (S2b)

func runImage(_ args: [String]) throws {
    guard args.count >= 3 else { throw SmokeError.usage("run <in.png> <out.png> <weightsDir> [--variant V] [--fp32] [--cpu] [--head N] [--tile N] [--overlap N] [--whole N] [--residual-fp32]") }
    let v = variant(args)
    let lane: HEART_Playback.Precision = flag(args, "--fp32") ? .fp32 : .fp16
    let headPad = Int(opt(args, "--head") ?? "64")!
    let cpu = flag(args, "--cpu")
    try withDevice(gpu: !cpu) {
        let tier = try HEART_Playback(variant: v, precision: lane, weightsDirectory: URL(fileURLWithPath: args[2]),
                                      wholeFrameMaxPixels: Int(opt(args, "--whole") ?? "\(HEART_Playback.defaultWholeFrameMaxPixels)")!,
                                      inputTileSize: Int(opt(args, "--tile") ?? "\(HEART_Playback.defaultInputTileSize)")!,
                                      tileOverlap: Int(opt(args, "--overlap") ?? "\(HEART_Playback.defaultTileOverlap)")!,
                                      attentionHeadPad: headPad, residualStreamFloat32: flag(args, "--residual-fp32"))
        let x = try loadRGB(args[0])
        let t0 = now()
        let y = try tier.forward(x); eval(y)
        let dt = now() - t0
        let t1 = now()
        let y2 = try tier.forward(x); eval(y2)
        let dt2 = now() - t1
        try savePNG(y, to: args[1])
        let peak = Double(MLX.Memory.snapshot().peakMemory) / 1_048_576
        print(String(format: "OK %@ %@ head %d ×%d %dx%d → %dx%d | first %.2f s, second %.2f s | peakGPU %.0f MB | %@",
                     v.rawValue, lane.rawValue, headPad, tier.scaleFactor, x.dim(2), x.dim(1), y.dim(2), y.dim(1), dt, dt2, peak, args[1]))
    }
}

// MARK: - engine (S7)

func runEngine(_ args: [String]) async throws {
    guard args.count >= 3 else { throw SmokeError.usage("engine <in.png> <out.png> <weightsDir> [--variant V] [--fp32] [--scale N] [--raw] [--whole N] [--tile N]") }
    let v = HEARTVariant(rawValue: opt(args, "--variant") ?? "fidelity") ?? .fidelity
    let quant: Quant = flag(args, "--fp32") ? .fp32 : .fp16
    let scale = opt(args, "--scale").map { Int($0)! }
    let cfg = HEARTConfiguration(variant: v, quant: quant,
                                 wholeFrameMaxPixels: opt(args, "--whole").map { Int($0)! },
                                 inputTileSize: opt(args, "--tile").map { Int($0)! },
                                 weightsDirectory: URL(fileURLWithPath: args[2]))
    if flag(args, "--cpu") { Device.setDefault(device: Device(.cpu)) }      // the whole engine run on the CPU stream
    let engine = MLXServeEngine()
    await engine.useModelStore(ModelStore(root: nil))
    let t0 = now()
    let id = try await engine.register(HEARTUpscalePackage.registration, configuration: cfg)
    let tReg = now() - t0
    var image = try loadPNGData(args[0])
    if flag(args, "--raw") {
        let pb = try HEARTUpscalePackage.decodeToPixelBuffer(image)
        image = HEARTUpscalePackage.encodeRawBGRA8(pb)!
    }
    let t1 = now()
    let resp = try await engine.run(ImageUpscaleRequest(image: image, scale: scale), package: id)
    let tRun = now() - t1
    guard let up = resp as? ImageUpscaleResponse else { throw SmokeError.badImage("response") }
    var outData = up.image.data
    if up.image.format == .rawBGRA8 {
        let pb = try HEARTUpscalePackage.rawBGRA8ToPixelBuffer(up.image)
        outData = HEARTUpscalePackage.encodePNG(pb)!
    }
    try outData.write(to: URL(fileURLWithPath: args[1]))
    let peak = Double(MLX.Memory.snapshot().peakMemory) / 1_048_576
    MLX.Memory.clearCache()
    let floorMB = Double(MLX.Memory.snapshot().activeMemory) / 1_048_576
    print(String(format: "OK engine %@ %@ ×%d → %dx%d %@ | reg %.2f s run %.2f s | MLX peak %.0f MB floor %.0f MB",
                 v.rawValue, quant.rawValue, up.appliedScale, up.image.width ?? -1, up.image.height ?? -1,
                 up.image.format.rawValue, tReg, tRun, peak, floorMB))
}

// MARK: - perf (H4)

func runPerf(_ args: [String]) throws {
    guard args.count >= 1 else { throw SmokeError.usage("perf <weightsDir> [--sizes 256,512] [--rounds 5] [--arms fp32-64,fp32-40,fp16-64]") }
    let sizes = (opt(args, "--sizes") ?? "256,512").split(separator: ",").map { Int($0)! }
    let rounds = Int(opt(args, "--rounds") ?? "5")!
    let armNames = (opt(args, "--arms") ?? "fp32-64,fp32-40,fp16-64,fp16-40").split(separator: ",").map(String.init)
    let gBefore = gpuUtilization()
    print("perf: GPU utilization before \(gBefore) % (sample the counter for ~10 s before starting; must be idle)")
    var arms: [(String, HEART)] = []
    for a in armNames {
        let parts = a.split(separator: "-"); let lane = HEART_Playback.Precision(rawValue: String(parts[0]))!
        let head = Int(parts[1])!
        arms.append((a, try makeModel(weightsDir: args[0], variant: .fidelity, precision: lane, headPad: head)))
    }
    var report: [String: Any] = ["gpu_before": gBefore, "rounds": rounds, "sizes": [String: Any]()]
    var sizesReport: [String: Any] = [:]
    for side in sizes {
        let x = MLXRandom.uniform(0 ..< 1, [1, side, side, 3]); eval(x)
        for (_, m) in arms { for _ in 0 ..< 2 { let y = m(x); eval(y) } }      // warm (compile + allocator)
        var t: [String: [Double]] = [:]
        for r in 0 ..< rounds {
            let order = Array(arms[(r % arms.count)...]) + Array(arms[..<(r % arms.count)])
            for (name, m) in order {
                let t0 = now(); let y = m(x); eval(y); let dt = (now() - t0) * 1000
                t[name, default: []].append(dt)
            }
        }
        let mpx = Double(side * 4 * side * 4) / 1e6
        print(String(format: "── LR %d² → %d² (%.2f Mpx out), median of %d, arms interleaved ──", side, side * 4, mpx, rounds))
        var sr: [String: Any] = [:]
        for (name, _) in arms {
            let s = t[name]!.sorted(); let med = s[s.count / 2]
            print(String(format: "  %-10@ %9.1f ms   %8.1f ms/Mpx-out   (all: %@)", name as NSString, med, med / mpx,
                         t[name]!.map { String(format: "%.0f", $0) }.joined(separator: ", ")))
            sr[name] = ["median_ms": med, "ms_per_mpx": med / mpx, "all_ms": t[name]!]
        }
        sizesReport["\(side)"] = sr
    }
    let gAfter = gpuUtilization()
    report["sizes"] = sizesReport; report["gpu_after"] = gAfter
    print("perf: GPU utilization after \(gAfter) %; RealPLKSR anchor 111 ms/Mpx (AB-R-0365), P0 bar 540–590 ms/Mpx at head 64 (AB-R-0370)")
    if let out = opt(args, "--json") {
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: out))
    }
}

// MARK: - bench (H3 dtype study + H5 tiling / memory, on the 27 bench cells)

/// An arm spec: `<lane>[r]-<headPad>-<geometry>` — lane `fp32` | `fp16` (`fp16r` = residual stream fp32), geometry
/// `whole` (single pass) or `t<tile>o<overlap>` (the shared tile driver). E.g. `fp32-64-whole`, `fp16-64-t384o32`.
struct ArmSpec {
    let name: String, lane: HEART_Playback.Precision, residualFP32: Bool, headPad: Int, tile: Int, overlap: Int, whole: Bool
    init(_ spec: String) throws {
        let parts = spec.split(separator: "-").map(String.init)
        guard parts.count == 3 else { throw SmokeError.usage("arm spec \(spec)") }
        name = spec
        residualFP32 = parts[0].hasSuffix("r")
        lane = HEART_Playback.Precision(rawValue: String(parts[0].prefix(4)))!
        headPad = Int(parts[1])!
        if parts[2] == "whole" { whole = true; tile = 512; overlap = 32 }
        else {
            let g = parts[2].dropFirst().split(separator: "o"); whole = false
            tile = Int(g[0])!; overlap = Int(g[1])!
        }
    }
    func tier(weights: URL, variant: HEART_Playback.Variant) throws -> HEART_Playback {
        try HEART_Playback(variant: variant, precision: lane, weightsDirectory: weights,
                           wholeFrameMaxPixels: whole ? .max : 0, inputTileSize: tile, tileOverlap: overlap,
                           attentionHeadPad: headPad, residualStreamFloat32: residualFP32)
    }
}

/// Run a tier on one BGRA pixel buffer synchronously (the CLI is single-threaded; the tier's `upscale` is async only
/// for the protocol).
func upscaleSync(_ tier: HEART_Playback, _ pb: CVPixelBuffer) throws -> CVPixelBuffer {
    let semaphore = DispatchSemaphore(value: 0)
    var result: CVPixelBuffer? = nil; var err: Error? = nil
    Task.detached { do { result = try await tier.upscale(pb) } catch { err = error }; semaphore.signal() }
    semaphore.wait()
    if let err { throw err }
    return result!
}

func runBench(_ args: [String]) throws {
    guard args.count >= 3 else {
        throw SmokeError.usage("bench <weightsDir> <benchDir> <outDir> [--arms a,b,…] [--variant V] [--json f] | bench <weightsDir> --memory-curve [--arms fp16-64-whole,fp32-64-whole]")
    }
    let weights = URL(fileURLWithPath: args[0])
    let v = variant(args)
    if flag(args, "--memory-curve") {
        let arms = try (opt(args, "--arms") ?? "fp16-64-whole,fp32-64-whole").split(separator: ",").map { try ArmSpec(String($0)) }
        print("whole-frame peak MLX memory (one forward each, fresh tier per size; GPU util before \(gpuUtilization()) %)")
        for arm in arms {
            for (w, h) in [(512, 512), (960, 540), (1280, 720), (1920, 1080)] {
                let tier = try arm.tier(weights: weights, variant: v)
                MLX.Memory.clearCache(); MLX.GPU.resetPeakMemory()
                let x = MLXRandom.uniform(0 ..< 1, [1, h, w, 3]); eval(x)
                let t0 = now(); let y = try tier.forward(x); eval(y); let dt = now() - t0
                let t1 = now(); let y2 = try tier.forward(x); eval(y2); let dt2 = now() - t1
                let peak = Double(MLX.Memory.snapshot().peakMemory) / 1_048_576
                print(String(format: "  %-14@ %dx%d → %dx%d: first %.2f s, second %.2f s, MLX peak %.0f MB", arm.name as NSString,
                             w, h, w * tier.scaleFactor, h * tier.scaleFactor, dt, dt2, peak))
            }
        }
        print("GPU util after \(gpuUtilization()) %")
        return
    }
    let benchDir = args[1], outDir = args[2]
    let arms = try (opt(args, "--arms") ?? "fp32-64-whole,fp16-64-whole,fp16r-64-whole,fp16-64-t256o32,fp16-64-t384o32,fp16-64-t512o32").split(separator: ",").map { try ArmSpec(String($0)) }
    try FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    // the scorer wants the references and lows beside the arms: copy them in (idempotent)
    let entries = try FileManager.default.contentsOfDirectory(atPath: benchDir)
    for f in entries where f.hasSuffix("__reference.png") || f.hasSuffix("__low.png") {
        let dst = "\(outDir)/\(f)"
        if !FileManager.default.fileExists(atPath: dst) { try FileManager.default.copyItem(atPath: "\(benchDir)/\(f)", toPath: dst) }
    }
    let lows = entries.filter { $0.hasSuffix("__low.png") }.sorted()
    let tiers = try arms.map { ($0, try $0.tier(weights: weights, variant: v)) }
    let gBefore = gpuUtilization()
    print("bench: \(lows.count) cells × \(arms.count) arms, variant \(v.rawValue), GPU util before \(gBefore) % (must be idle)")
    // warm every arm once (compile / allocator) on the first cell
    if let first = lows.first {
        let pb = try HEARTUpscalePackage.decodeToPixelBuffer(try loadPNGData("\(benchDir)/\(first)"))
        for (_, tier) in tiers { _ = try upscaleSync(tier, pb) }
    }
    var times: [String: [Double]] = [:], peaks: [String: [Double]] = [:], dbs: [String: [Float]] = [:]
    var perCell: [[String: Any]] = []
    for (i, low) in lows.enumerated() {
        let stem = String(low.dropLast("__low.png".count))
        let pb = try HEARTUpscalePackage.decodeToPixelBuffer(try loadPNGData("\(benchDir)/\(low)"))
        var reference: MLXArray? = nil
        var line = "  \(i + 1)/\(lows.count) \(stem):"
        var row: [String: Any] = ["cell": stem]
        // rotate the arm order per cell so no arm always follows the same predecessor
        let order = Array(tiers[(i % tiers.count)...]) + Array(tiers[..<(i % tiers.count)])
        var outputs: [String: MLXArray] = [:]
        for (arm, tier) in order {
            MLX.GPU.resetPeakMemory()
            let t0 = now()
            let out = try upscaleSync(tier, pb)
            let dt = now() - t0
            let peak = Double(MLX.Memory.snapshot().peakMemory) / 1_048_576
            let png = HEARTUpscalePackage.encodePNG(out)!
            let path = "\(outDir)/\(stem)__HEART-\(arm.name).png"
            try png.write(to: URL(fileURLWithPath: path))
            outputs[arm.name] = try loadRGB(path)
            times[arm.name, default: []].append(dt); peaks[arm.name, default: []].append(peak)
            row[arm.name] = ["s": dt, "peakMB": peak]
        }
        reference = outputs[arms[0].name]
        for arm in arms {
            let y = outputs[arm.name]!
            if arm.name == arms[0].name {
                line += String(format: " %@ %.2fs/%.0fMB", arm.name as NSString, times[arm.name]!.last!, peaks[arm.name]!.last!)
            } else {
                let p = psnr(y, reference!); dbs[arm.name, default: []].append(p)
                line += String(format: " %@ %.1fdB/%.2fs/%.0fMB", arm.name as NSString, p, times[arm.name]!.last!, peaks[arm.name]!.last!)
                var d = row[arm.name] as! [String: Any]; d["dB_vs_\(arms[0].name)"] = p; row[arm.name] = d
            }
        }
        perCell.append(row)
        print(line)
    }
    let gAfter = gpuUtilization()
    print("── per arm over \(lows.count) cells (8-bit PNGs; dB = PSNR vs the \(arms[0].name) arm) ──")
    var summary: [String: Any] = [:]
    for arm in arms {
        let t = times[arm.name]!.sorted(), pk = peaks[arm.name]!.sorted()
        var s = String(format: "  %-16@ median %.2f s  peak %.0f MB", arm.name as NSString, t[t.count / 2], pk[pk.count / 2])
        var d: [String: Any] = ["median_s": t[t.count / 2], "median_peakMB": pk[pk.count / 2]]
        if let db = dbs[arm.name]?.sorted() {
            s += String(format: "  vs %@: median %.1f dB  min %.1f  max %.1f", arms[0].name as NSString, db[db.count / 2], db.first!, db.last!)
            d["dB_median"] = db[db.count / 2]; d["dB_min"] = db.first!; d["dB_max"] = db.last!
        }
        print(s); summary[arm.name] = d
    }
    print("GPU util before \(gBefore) % after \(gAfter) %. Score with: vosrgate score \(outDir) <out.csv> (SSIMULACRA2 vs __reference.png)")
    if let out = opt(args, "--json") {
        let report: [String: Any] = ["gpu_before": gBefore, "gpu_after": gAfter, "arms": arms.map(\.name), "summary": summary, "cells": perCell]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: out))
    }
}

// MARK: - cancel (live mid-run probe)

func runCancel(_ args: [String]) async throws {
    guard args.count >= 1 else { throw SmokeError.usage("cancel <weightsDir> [--after 1.5] [--side 1024]") }
    let after = Double(opt(args, "--after") ?? "1.5")!
    let side = Int(opt(args, "--side") ?? "1024")!
    let cfg = HEARTConfiguration(variant: .fidelity, quant: .fp16, wholeFrameMaxPixels: 0, inputTileSize: 512,
                                 weightsDirectory: URL(fileURLWithPath: args[0]))
    let engine = MLXServeEngine()
    await engine.useModelStore(ModelStore(root: nil))
    let id = try await engine.register(HEARTUpscalePackage.registration, configuration: cfg)
    let x = MLXRandom.uniform(0 ..< 1, [1, side, side, 3]); eval(x)
    let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("heart-cancel-\(UUID().uuidString).png").path
    try savePNG(x, to: tmp)
    let image = try loadPNGData(tmp)
    _ = try await engine.prepare(.imageUpscale, package: id)
    let t0 = now()
    let full = try await engine.run(ImageUpscaleRequest(image: image), package: id)
    let tFull = now() - t0
    _ = full
    let task = Task { try await engine.run(ImageUpscaleRequest(image: image), package: id) }
    try await Task.sleep(nanoseconds: UInt64(after * 1e9))
    let tc = now()
    task.cancel()
    var outcome = "returned a result (NOT cancelled)"
    do { _ = try await task.value } catch is CancellationError { outcome = "CancellationError (unwrapped)" } catch { outcome = "other error: \(error)" }
    let latency = now() - tc
    print(String(format: "cancel probe: full run %.2f s; cancelled at %.2f s → %@ after %.3f s", tFull, after, outcome, latency))
}

// MARK: - main

let argv = Array(CommandLine.arguments.dropFirst())
guard let mode = argv.first else {
    log("usage: heart-smoke keys|gate|run|engine|perf|bench|cancel …"); exit(2)
}
let rest = Array(argv.dropFirst())
do {
    switch mode {
    case "keys": try runKeys(rest)
    case "gate": try runGate(rest)
    case "run": try runImage(rest)
    case "engine": try await runEngine(rest)
    case "perf": try runPerf(rest)
    case "bench": try runBench(rest)
    case "cancel": try await runCancel(rest)
    default: throw SmokeError.usage("unknown mode \(mode)")
    }
} catch {
    log("FAILED: \(error)")
    exit(1)
}
exit(0)
