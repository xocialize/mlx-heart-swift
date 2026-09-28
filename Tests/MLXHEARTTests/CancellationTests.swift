//
//  CancellationTests.swift — the offline CAN gate (engine ≥ 0.27.0): CAN-1/CAN-2 pre-cancelled run() propagation +
//  classification, and CAN-3 the checkpoint-cadence declaration of record. The live mid-run probe (type + latency)
//  is `heart-smoke cancel`.
//

import XCTest
import Foundation
import MLXToolKit
import MLXServeConformance
@testable import MLXHEART

final class CancellationTests: XCTestCase {

    func testCANGatePreCancelledRun() async {
        // Construction is cheap (C13) and the entry checkpoint throws before validation, decode, or the weights
        // are touched, so this is offline-safe.
        let package = HEARTUpscalePackage(configuration: HEARTConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: ImageUpscaleRequest(image: Image(format: .png, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        // peakActivationBytes ≥ 2 GB ⇒ long-run implied; the sub-second exemption is not available (a whole-frame
        // 4K output runs ~4.5–4.9 s on M5 Max).
        XCTAssertTrue(CancellationConformance.longRunImplied(by: HEARTUpscalePackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: HEARTUpscalePackage.manifest,
            posture: .cadence([
                // The shared tile driver checks Task.checkCancellation once per tile (top of the tile loop)
                // ("chunk" = one tile), AND the core evaluates + checks after every residual group of every
                // forward — six per tile / whole frame ("layer" = one RHAG) — so a cancel lands within ~1/6 of a
                // forward even on the whole-frame path. RunProgress reports at both seams (step = tile·groups +
                // group, stage = group).
                .init(phase: .upsample, unit: .chunk, reportsRunProgress: true),
                .init(phase: .upsample, unit: .layer, reportsRunProgress: true),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
