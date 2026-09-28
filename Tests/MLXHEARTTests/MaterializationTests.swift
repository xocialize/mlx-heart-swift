//
//  MaterializationTests.swift — the offline MAT gate (engine ≥ 0.19.0, contract 1.24 engine-executed
//  materialization) for a NETWORK-sourced package: MAT-1 store-stampable, MAT-2 sources declared, MAT-3 role/repo
//  hygiene, MAT-4 fresh-machine posture (nil store ⇒ the variant's file is missing), MAT-5 explicit paths satisfy —
//  per variant × lane, because the declaration changes with both.
//

import XCTest
import Foundation
import MLXToolKit
import MLXServeCore
import MLXServeConformance
@testable import MLXHEART

final class MaterializationTests: XCTestCase {

    private func satisfiedDirectory(for variant: HEARTVariant, quant: Quant) throws -> URL {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("heart-mat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        for f in HEARTConfiguration.files(for: variant, quant: quant) {
            FileManager.default.createFile(atPath: tmp.appendingPathComponent(f).path, contents: Data([0]))
        }
        return tmp
    }

    func testFullMATGatePassesPerVariantAndLane() throws {
        for v in HEARTVariant.allCases {
            for quant in [Quant.fp16, .fp32] {
                let dir = try satisfiedDirectory(for: v, quant: quant)
                defer { try? FileManager.default.removeItem(at: dir) }
                let report = MaterializationConformance.check(
                    freshConfiguration: HEARTConfiguration(variant: v, quant: quant),
                    satisfiedConfiguration: HEARTConfiguration(variant: v, quant: quant, weightsDirectory: dir))
                XCTAssertTrue(report.passed, "\(v) \(quant):\n\(report.summary)")
            }
        }
    }

    /// The store's flat layout (contract 1.24: files directly under `models--mlx-community--HEART-fp16/`) satisfies
    /// the variant — and only when ITS file is there (another variant's file does not count).
    func testStoreFlatLayoutSatisfiesOnlyTheVariantsOwnFile() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("heart-store-\(UUID().uuidString)")
        let repoDir = root.appendingPathComponent(ModelStore.repoFolderName(for: HEARTConfiguration.fp16Repo))
        try FileManager.default.createDirectory(at: repoDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let fid = HEARTConfiguration(variant: .fidelity, quant: .fp16, modelsRootDirectory: root)
        let sharp = HEARTConfiguration(variant: .sharp, quant: .fp16, modelsRootDirectory: root)
        XCTAssertEqual(fid.missingWeightSources(storeRoot: root).count, 1, "empty repo dir ⇒ missing")
        FileManager.default.createFile(atPath: repoDir.appendingPathComponent("config.json").path, contents: Data([0]))
        FileManager.default.createFile(atPath: repoDir.appendingPathComponent("heart_4x_otf_v2_fp16.safetensors").path, contents: Data([0]))
        XCTAssertTrue(fid.missingWeightSources(storeRoot: root).isEmpty)
        XCTAssertEqual(sharp.missingWeightSources(storeRoot: root).count, 1, "the other variant is still missing")
        XCTAssertEqual(fid.resolvedWeightsDirectory(storeRoot: root)?.standardizedFileURL.path, repoDir.standardizedFileURL.path)
        XCTAssertEqual(fid.prewarmPaths.count, 2)
        XCTAssertTrue(fid.prewarmPaths.allSatisfy { FileManager.default.fileExists(atPath: $0.path) })
        // the fp32 lane lives in its own repo folder
        let fid32 = HEARTConfiguration(variant: .fidelity, quant: .fp32, modelsRootDirectory: root)
        XCTAssertEqual(fid32.missingWeightSources(storeRoot: root).count, 1)
    }

    /// Register is offline — a fresh registration with no store reports a needed download.
    func testEngineNeedsDownloadOnAFreshRegistration() async throws {
        let engine = MLXServeEngine()
        _ = try await engine.register(HEARTUpscalePackage.registration, configuration: HEARTConfiguration())
        let needs = await engine.needsDownload(.imageUpscale)
        XCTAssertTrue(needs, "nothing materialized, the variant must read as needing a download")
    }
}
