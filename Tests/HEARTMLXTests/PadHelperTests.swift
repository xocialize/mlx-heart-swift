//
//  PadHelperTests.swift — `conv_before_upsample` (embedDim → numFeat, 180 → 64, 3×3, stride 1) is the shape mlx 0.32.2
//  serves with `pad_in_channels_conv_2D_gpu`, which frees its zero fill value while the GPU still reads it (AB-T-0214).
//  HEART builds it as a `RoutedConv2d` (route `.exact`), so ExactConv pads the channels in the graph; on the GPU it
//  must still match a plain conv on the CPU.
//

import Foundation
import MLX
import MLXExactConv
import MLXNN
import XCTest
@testable import HEARTMLX

final class PadHelperTests: XCTestCase {
    func testConvBeforeUpsampleIsRoutedOffThePadHelper() {
        let m = HEART(config: HEARTConfig())
        guard let conv = m.convBeforeUpsample[0] as? RoutedConv2d else {
            return XCTFail("conv_before_upsample is not routed")
        }
        XCTAssertEqual(conv.route, .exact)
        MLXRandom.seed(5)
        let params = conv.parameters().mapValues { MLXRandom.normal($0.shape) * 0.05 }
        let x = MLXRandom.normal([1, 48, 40, 180])
        XCTAssertTrue(
            ExactConv.takesPadHelper2D(input: x, weight: conv.weight, padding: (1, 1)),
            "a plain Conv2d at this shape would take mlx's pad helper")

        for dtype in [DType.float32, .float16] {
            conv.update(parameters: params.mapValues { $0.asType(dtype) })
            let xd = x.asType(dtype)
            let gpu = Device.withDefaultDevice(Device(.gpu)) { conv(xd) }
            let cpu = Device.withDefaultDevice(Device(.cpu)) {
                conv2d(xd, conv.weight, padding: 1) + conv.bias!
            }
            eval(gpu, cpu)
            let err = abs(gpu.asType(.float32) - cpu.asType(.float32)).max().item(Float.self)
            XCTAssertLessThan(err, dtype == .float32 ? 1e-4 : 2e-2, "\(dtype): \(err)")
        }
    }
}
