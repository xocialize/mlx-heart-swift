//
//  HEARTCoreTests.swift — core tests that need no weights: the parameter-tensor counts pinned to the checkpoint
//  inventory (748 / 746), the parameter counts, inference mode (C14), the construction-time block schedule
//  (attention on even blocks, shift on odd attention indices), the op substitutions against hand-computed
//  expectations (reflect / replicate padding, `check_image_size`, pixel shuffle), the RIB coordinate table, and the
//  two SHAPE-SAFE porting mistakes as failing probes (the `(heads, 3, c)` QKV read; the `(r, r, C)` pixel shuffle).
//  Plus the S0 key contract when `HEART_WEIGHTS` points at the converter's output directory.
//
//  Everything runs on the CPU stream under XCTest: the swift-testing runner's dlopen'ed bundle cannot see mlx's
//  metallib (every MLX call dies, CPU stream included), while XCTest's NSBundle-loaded .xctest can.
//

import Foundation
import XCTest
import MLX
import MLXNN
@testable import HEARTMLX

private func withCPU<R>(_ body: () throws -> R) rethrows -> R {
    try Device.withDefaultDevice(Device(.cpu), body)
}

private func totalParameterCount(_ module: Module) -> Int {
    module.parameters().flattened().reduce(0) { $0 + $1.1.size }
}

/// A small config that still exercises every block kind: one group of four blocks → attention on 0 (plain) and
/// 2 (shifted), none on 1 and 3.
private func smallConfig(scale: Int = 4) -> HEARTConfig {
    var c = HEARTConfig(scale: scale)
    c.depths = [4]; c.numHeads = [6]
    return c
}

final class HEARTCoreTests: XCTestCase {

    // MARK: structure

    func testTensorAndParameterCountsMatchTheCheckpointInventory() {
        withCPU {
            let x4 = HEART(config: HEARTConfig(scale: 4))
            XCTAssertEqual(x4.parameters().flattened().count, HEART.tensorCountX4)
            XCTAssertEqual(totalParameterCount(x4), HEART.parameterCountX4)
            let x2 = HEART(config: HEARTConfig(scale: 2))
            XCTAssertEqual(x2.parameters().flattened().count, HEART.tensorCountX2)
            XCTAssertEqual(totalParameterCount(x2), HEART.parameterCountX2)
            XCTAssertEqual(x4.attentionBlocks.count, 18)
        }
    }

    func testExpectedKeysAreTheUpstreamStateDictKeys() {
        withCPU {
            let keys = HEART.expectedKeys(scale: 4)
            XCTAssertEqual(keys.count, HEART.tensorCountX4)
            for k in ["conv_first.weight", "patch_embed.norm.weight", "norm.bias", "conv_after_body.bias",
                      "conv_before_upsample.0.weight", "upsample.0.weight", "upsample.2.bias", "conv_last.weight",
                      "layers.0.residual_group.blocks.0.norm1.weight",
                      "layers.0.residual_group.blocks.0.attn.to_qkv.weight",
                      "layers.0.residual_group.blocks.0.attn.to_hidden",
                      "layers.0.residual_group.blocks.0.attn.hidden_b",
                      "layers.0.residual_group.blocks.0.attn.to_q",
                      "layers.0.residual_group.blocks.0.attn.to_k",
                      "layers.0.residual_group.blocks.0.conv_block.cab.0.weight",
                      "layers.0.residual_group.blocks.0.conv_block.cab.2.bias",
                      "layers.0.residual_group.blocks.0.conv_block.cab.3.attention.1.weight",
                      "layers.0.residual_group.blocks.0.conv_block.cab.3.attention.3.bias",
                      "layers.0.residual_group.blocks.0.mlp.fc1.weight",
                      "layers.5.residual_group.blocks.5.mlp.fc2.bias",
                      "layers.5.conv.weight"] {
                XCTAssertTrue(keys.contains(k), "missing \(k)")
            }
            // odd blocks carry no attention keys; the shuffle placeholders and the LeakyReLU carry nothing
            XCTAssertFalse(keys.contains("layers.0.residual_group.blocks.1.attn.to_qkv.weight"))
            XCTAssertFalse(keys.contains { $0.hasPrefix("upsample.1.") || $0.hasPrefix("conv_before_upsample.1.") })
            XCTAssertFalse(HEART.expectedKeys(scale: 2).contains("upsample.2.weight"))
        }
    }

    /// Born in inference mode (C14) at the single construction choke point; the gate can fail (train(true)).
    func testInferenceMode() {
        withCPU {
            let m = HEART(config: smallConfig())
            var flags: [(String, Bool)] = []
            m.visit(modules: { flags.append(($0, $1.training)) })
            XCTAssertFalse(flags.isEmpty)
            XCTAssertTrue(flags.allSatisfy { !$0.1 }, "modules in training mode: \(flags.filter { $0.1 }.map(\.0))")
            m.train(true)
            var after: [(String, Bool)] = []
            m.visit(modules: { after.append(($0, $1.training)) })
            XCTAssertTrue(after.contains { $0.1 }, "the check must be able to fail")
        }
    }

    /// Block type is decided at construction: attention on `i % 2 == 0`, shift on odd attention indices per group.
    func testBlockScheduleIsFixedAtConstruction() {
        withCPU {
            let m = HEART(config: HEARTConfig(scale: 4))
            for layer in m.layers {
                let blocks = layer.residualGroup.blocks
                XCTAssertEqual(blocks.count, 6)
                XCTAssertEqual(blocks.map { $0.attn != nil }, [true, false, true, false, true, false])
                XCTAssertEqual(blocks.map(\.shift), [false, false, true, false, false, false])
                XCTAssertEqual(blocks.compactMap { $0.attn?.shift }, [false, true, false])
            }
        }
    }

    func testScaleIsDerivedFromTheUpsampleConvs() {
        XCTAssertEqual(HEART.scale(ofCheckpointKeys: ["upsample.0.weight", "upsample.2.weight", "conv_last.weight"]), 4)
        XCTAssertEqual(HEART.scale(ofCheckpointKeys: ["upsample.0.weight", "upsample.0.bias"]), 2)
        XCTAssertEqual(HEARTConfig(scale: 4).upsampleStages, 2)
        XCTAssertEqual(HEARTConfig(scale: 2).upsampleStages, 1)
        XCTAssertEqual(HEARTConfig(scale: 8).upsampleStages, 3)
    }

    // MARK: forward shapes

    func testForwardShapesIncludingWindowPadding() throws {
        try withCPU {
            let m = HEART(config: smallConfig())
            let y = m(MLXArray.zeros([1, 64, 64, 3])); eval(y)
            XCTAssertEqual(y.shape, [1, 256, 256, 3])
            // 20×20: pad 12 < 20 → reflect; 12×12: pad 20 ≥ 12 → replicate; both crop back to h·4 × w·4
            let y20 = m(MLXArray.zeros([1, 20, 20, 3])); eval(y20)
            XCTAssertEqual(y20.shape, [1, 80, 80, 3])
            let y12 = m(MLXArray.zeros([1, 12, 12, 3])); eval(y12)
            XCTAssertEqual(y12.shape, [1, 48, 48, 3])
            let odd = m(MLXArray.zeros([1, 40, 70, 3])); eval(odd)
            XCTAssertEqual(odd.shape, [1, 160, 280, 3])
            let x2 = HEART(config: smallConfig(scale: 2))
            let y2 = x2(MLXArray.zeros([1, 32, 32, 3])); eval(y2)
            XCTAssertEqual(y2.shape, [1, 64, 64, 3])
            // the seam form evaluates per group and reports progress
            var groups: [Int] = []
            let ys = try m.forward(MLXArray.zeros([1, 32, 32, 3]), checkpoint: {}) { g, _ in groups.append(g) }
            eval(ys)
            XCTAssertEqual(groups, [1])
        }
    }

    // MARK: ops

    func testReflectAndReplicatePadMatchTorchSemantics() {
        withCPU {
            // rows 0..3, cols 0..4, one channel: value = 10·row + col
            var v = [Float](); for r in 0 ..< 4 { for c in 0 ..< 5 { v.append(Float(10 * r + c)) } }
            let x = MLXArray(v, [1, 4, 5, 1])
            let r = reflectPad(x, top: 2, bottom: 1, left: 1, right: 3)
            XCTAssertEqual(r.shape, [1, 7, 9, 1])
            let rr = r.reshaped([7, 9]).asArray(Float.self)
            // torch reflect: top rows are x[2], x[1]; bottom row is x[2]; left col is x[.,1]; right cols x[.,3], x[.,2], x[.,1]
            XCTAssertEqual(Array(rr[0 ..< 9]), [21, 20, 21, 22, 23, 24, 23, 22, 21])
            XCTAssertEqual(Array(rr[9 ..< 18]), [11, 10, 11, 12, 13, 14, 13, 12, 11])
            XCTAssertEqual(Array(rr[18 ..< 27]), [1, 0, 1, 2, 3, 4, 3, 2, 1])
            XCTAssertEqual(Array(rr[54 ..< 63]), [21, 20, 21, 22, 23, 24, 23, 22, 21])
            let e = replicatePad(x, top: 0, bottom: 2, left: 0, right: 1)
            XCTAssertEqual(e.shape, [1, 6, 6, 1])
            let ee = e.reshaped([6, 6]).asArray(Float.self)
            XCTAssertEqual(Array(ee[0 ..< 6]), [0, 1, 2, 3, 4, 4])
            XCTAssertEqual(Array(ee[30 ..< 36]), [30, 31, 32, 33, 34, 34])
            // check_image_size: 20 → 32 reflect (pad 12 < 20); 12 → 32 replicate (pad 20 ≥ 12); aligned → identity
            let x20 = MLXArray((0 ..< 400).map { Float($0) }, [1, 20, 20, 1])
            let p20 = checkImageSize(x20, window: 32); XCTAssertEqual(p20.shape, [1, 32, 32, 1])
            XCTAssertEqual(p20[0, 20, 0, 0].item(Float.self), x20[0, 18, 0, 0].item(Float.self))   // reflect: row 20 ← row 18
            let x12 = MLXArray((0 ..< 144).map { Float($0) }, [1, 12, 12, 1])
            let p12 = checkImageSize(x12, window: 32); XCTAssertEqual(p12.shape, [1, 32, 32, 1])
            XCTAssertEqual(p12[0, 31, 0, 0].item(Float.self), x12[0, 11, 0, 0].item(Float.self))   // replicate: edge repeats
            let aligned = MLXArray.zeros([1, 64, 96, 3])
            XCTAssertEqual(checkImageSize(aligned, window: 32).shape, [1, 64, 96, 3])
        }
    }

    /// `pixelShuffleNHWC` matches torch's index map; the `(r, r, C)` reading is shape-identical and WRONG.
    func testPixelShuffleMatchesTorchAndTheWrongOrderingFailsLoudly() {
        withCPU {
            let (h, w, c, r) = (2, 3, 3, 2)
            let x = MLXArray((0 ..< (h * w * c * r * r)).map { Float($0) }, [1, h, w, c * r * r])
            let y = pixelShuffleNHWC(x, r); eval(y)
            XCTAssertEqual(y.shape, [1, h * r, w * r, c])
            // torch: out[c, hh·r + i, ww·r + j] = in[c·r·r + i·r + j, hh, ww]
            let xin = x.reshaped([h, w, c * r * r]).asArray(Float.self)
            let yout = y.reshaped([h * r, w * r, c]).asArray(Float.self)
            for hh in 0 ..< h { for ww in 0 ..< w { for cc in 0 ..< c { for i in 0 ..< r { for j in 0 ..< r {
                let expect = xin[(hh * w + ww) * (c * r * r) + cc * r * r + i * r + j]
                let got = yout[((hh * r + i) * (w * r) + (ww * r + j)) * c + cc]
                XCTAssertEqual(got, expect)
            }}}}}
            let wrong = x.reshaped([1, h, w, r, r, c]).transposed(0, 1, 3, 2, 4, 5).reshaped([1, h * r, w * r, c])
            eval(wrong)
            XCTAssertEqual(wrong.shape, y.shape)
            XCTAssertGreaterThan(abs(wrong - y).max().item(Float.self), 1)
        }
    }

    /// The RIB coordinate table: normalised (x, y) with the Fourier features in upstream's column order.
    func testRIBCoordinateTable() {
        withCPU {
            let t = RIBCoordinates.shared(window: 32, nFreqs: 10)
            XCTAssertEqual(t.coords.shape, [1024, 42])
            let row = t.coords[33].asArray(Float.self)          // p = 33 → y = 1, x = 1
            let v = Float(2.0 * (1.0 + 0.5) / 32.0 - 1.0)
            XCTAssertEqual(row[0], v); XCTAssertEqual(row[1], v)
            XCTAssertEqual(row[2], Float(sin(Double(v)))); XCTAssertEqual(row[3], Float(sin(Double(v))))
            XCTAssertEqual(row[4], Float(cos(Double(v)))); XCTAssertEqual(row[5], Float(cos(Double(v))))
            XCTAssertEqual(row[2 + 4 * 9], Float(sin(Double(v * 512))), accuracy: 1e-7)
            let row1 = t.coords[1].asArray(Float.self)          // p = 1 → x = 1, y = 0
            XCTAssertEqual(row1[0], v); XCTAssertEqual(row1[1], Float(2.0 * 0.5 / 32.0 - 1.0))
            XCTAssertTrue(RIBCoordinates.shared(window: 32, nFreqs: 10) === t)
        }
    }

    // MARK: the shape-safe traps

    /// The channel-attention pool reduces in fp32: a half-precision tile whose spatial sum exceeds 65504 must not
    /// overflow (the first fp16 run went NaN here — an fp16 accumulation of 4096 × |70|).
    func testChannelAttentionPoolDoesNotOverflowInHalf() {
        withCPU {
            let ca = ChannelAttention(numFeat: 180, squeezeFactor: 30)
            let x = (MLXArray.ones([1, 64, 64, 180]) * Float(70)).asType(.float16)   // a Float scalar operand promotes fp16 → fp32
            let y = ca(x); eval(y)
            XCTAssertEqual(y.dtype, .float16)
            XCTAssertEqual(sum(logicalNot(isFinite(y))).item(Int32.self), 0, "NaN/inf in the half-precision channel attention")
            // the pooled mean of a constant tile is the constant, so the gate equals sigmoid(conv(conv(70))) ∈ [0, 1]
            let gate = y.asType(.float32)[0, 0, 0, 0].item(Float.self) / 70
            XCTAssertTrue(gate >= 0 && gate <= 1, "gate \(gate)")
        }
    }

    /// Reading the 3·dim projection as `(heads, 3, c)` gives the same shapes and a different answer.
    func testQKVOrderProbeDiscriminates() {
        withCPU {
            let attn = RIBWindowAttention(dim: 180, windowSize: 32, numHeads: 6, rank: 8, ribHiddenDim: 32, ribNFreqs: 10,
                                          shift: false, headPad: 64)
            MLXRandom.seed(7)
            attn.update(parameters: attn.parameters().mapValues { MLXRandom.normal($0.shape) * 0.05 })
            let x = MLXRandom.normal([1, 32, 32, 180])
            let good = attn(x); eval(good)
            // the wrong reading, spelled out on the same weights
            let hd = 30, heads = 6, n = 1024
            let qkvW = attn.toQKV(x).reshaped([1, 1, 32, 1, 32, heads, 3, hd])
                .transposed(6, 0, 1, 3, 5, 2, 4, 7).reshaped([3, 1, heads, n, hd])
            let q = qkvW[0] * pow(Float(hd), -0.5), k = qkvW[1], v = qkvW[2]
            let f = attn.prepared(dtype: .float32)
            let qc = concatenated([q, f.qExtra], axis: -1), kc = concatenated([k, f.kExtra], axis: -1)
            let vc = concatenated([v, f.vExtra], axis: -1)
            let o = MLXFast.scaledDotProductAttention(queries: qc, keys: kc, values: vc, scale: 1, mask: nil)[0..., 0..., 0..., 0 ..< hd]
            let wrong = attn.toOut(o.reshaped([1, 1, 1, heads, 32, 32, hd]).transposed(0, 1, 4, 2, 5, 3, 6).reshaped([1, 32, 32, 180]))
            eval(wrong)
            XCTAssertEqual(wrong.shape, good.shape)
            let rel = abs(wrong - good).max().item(Float.self) / abs(good).max().item(Float.self)
            XCTAssertGreaterThan(rel, 0.1, "the (heads, 3, c) reading must be loudly different (rel \(rel))")
        }
    }

    /// Zero-padding the (q ‖ pos) head from 40 to 64 columns is identical math (the CPU path is unfused on both).
    func testHeadPad64EqualsHeadPad40() {
        withCPU {
            MLXRandom.seed(11)
            let a = RIBWindowAttention(dim: 180, windowSize: 32, numHeads: 6, rank: 8, ribHiddenDim: 32, ribNFreqs: 10, shift: true, headPad: 40)
            let b = RIBWindowAttention(dim: 180, windowSize: 32, numHeads: 6, rank: 8, ribHiddenDim: 32, ribNFreqs: 10, shift: true, headPad: 64)
            let params = a.parameters().mapValues { MLXRandom.normal($0.shape) * 0.05 }
            a.update(parameters: params); b.update(parameters: params)
            let x = MLXRandom.normal([1, 64, 32, 180])
            let ya = a(x), yb = b(x); eval(ya, yb)
            let rel = abs(ya - yb).max().item(Float.self) / abs(ya).max().item(Float.self)
            XCTAssertLessThan(rel, 1e-5, "rel \(rel)")
        }
    }

    /// The shifted block attends over a 16-px reflect pad on every side and crops back — never a cyclic roll.
    func testShiftedAttentionPadsAndCropsToTheInputSize() {
        withCPU {
            let a = RIBWindowAttention(dim: 180, windowSize: 32, numHeads: 6, rank: 8, ribHiddenDim: 32, ribNFreqs: 10, shift: true, headPad: 64)
            let x = MLXRandom.normal([1, 64, 96, 180])
            XCTAssertEqual(a.shiftPadded(x).shape, [1, 96, 128, 180])
            let y = a(x); eval(y)
            XCTAssertEqual(y.shape, [1, 64, 96, 180])
        }
    }

    // MARK: S0 (env-gated: HEART_WEIGHTS=<converter output dir>)

    func testS0KeyContractOnConvertedWeights() throws {
        guard let dir = ProcessInfo.processInfo.environment["HEART_WEIGHTS"] else {
            throw XCTSkip("set HEART_WEIGHTS to run the S0 key contract on real files")
        }
        try withCPU {
            for v in HEART_Playback.Variant.allCases {
                for lane in HEART_Playback.Precision.allCases {
                    let url = URL(fileURLWithPath: dir).appendingPathComponent(v.fileName(precision: lane))
                    let arrays = try MLX.loadArrays(url: url)
                    let scale = HEART.scale(ofCheckpointKeys: arrays.keys)
                    XCTAssertEqual(scale, v.scale)
                    XCTAssertEqual(Set(arrays.keys), HEART.expectedKeys(scale: scale), "\(url.lastPathComponent)")
                    let m = HEART(config: HEARTConfig(scale: scale))
                    XCTAssertNoThrow(try m.loadWeights(arrays), url.lastPathComponent)
                    XCTAssertEqual(m.computeDtype, lane == .fp16 ? .float16 : .float32)
                    // a partial checkpoint is refused loudly
                    var partial = arrays; partial.removeValue(forKey: "conv_last.bias")
                    XCTAssertThrowsError(try HEART(config: HEARTConfig(scale: scale)).loadWeights(partial))
                }
            }
        }
    }
}
