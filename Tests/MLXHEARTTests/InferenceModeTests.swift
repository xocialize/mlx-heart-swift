//
//  InferenceModeTests.swift — the C14 INF gate (engine ≥ 0.36.0 / contract 1.27.0) on the core graph the package
//  loads. HEART carries no BatchNorm / running statistics (i-LN and AffineTransform are stateless), so the gate's
//  bite here is the general one: every module in the loaded tree reports `training == false`, set at the single
//  construction choke point (`HEART.init`). The load-bearing assertion is the inversion — the gate FAILS on a graph
//  flipped back to training mode — so deleting the choke point cannot leave the suite green.
//

import XCTest
import Foundation
import MLX
import MLXServeConformance
import MLXServeConformanceNN
import HEARTMLX
@testable import MLXHEART

final class InferenceModeTests: XCTestCase {

    private func withCPU<R>(_ body: () throws -> R) rethrows -> R {
        try Device.withDefaultDevice(Device(.cpu), body)
    }

    func testINFGatePassesOnAConstructedGraphAndCanFail() {
        withCPU {
            var cfg = HEARTConfig(scale: 4); cfg.depths = [2]; cfg.numHeads = [6]
            let model = HEART(config: cfg)
            let flags = InferenceModeConformance.flags(of: ["model": model])
            XCTAssertFalse(flags.isEmpty)
            let pass = InferenceModeConformance.check(flags: flags, posture: .moduleGraph)
            XCTAssertTrue(pass.passed, pass.summary)
            model.train(true)
            let fail = InferenceModeConformance.check(flags: InferenceModeConformance.flags(of: ["model": model]),
                                                      posture: .moduleGraph)
            XCTAssertFalse(fail.passed, "the gate must be able to fail")
        }
    }

    /// The package's seam exposes the loaded graph by role (nil until `load()`), so the live gate has something to walk.
    func testPackageExposesTheGraphByRole() async {
        let pkg = HEARTUpscalePackage(configuration: HEARTConfiguration())
        let graphs = await pkg.inferenceModeGraphs
        XCTAssertEqual(Array(graphs.keys), ["model"])
        XCTAssertNil(graphs["model"] ?? nil)
    }
}
