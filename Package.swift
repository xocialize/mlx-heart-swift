// swift-tools-version: 6.2
import PackageDescription

// mlx-heart-swift — HEART (Phips/HEART, Apache-2.0; Philip Hofmann's HAT-iLN + RIB window-attention 4× super-resolution,
// 16.7 M params, trained only on the CC0 LUCID corpus) ported PyTorch → Swift/MLX for MLXEngine as the fidelity-class
// stills upscale tier (AB-T-0189; plan mlxengine-forge/Docs/NERVE-HEART-PORT-PLAN.md §3). ONE repo, TWO products, the
// mlx-realplksr-swift shape:
//   • HEARTMLX — engine-agnostic Swift/MLX core, isomorphic to upstream `heart_arch.py` + the traiNNer-redux helpers it
//     imports (i-LN, CAB, Mlp, AffineTransform, Upsample): NHWC, per-group cancellation checkpoints, a `PlaybackTier`
//     over the SHARED tile driver from RealESRGANMLX. No MLXToolKit dependency.
//   • MLXHEART — the MLXEngine `imageUpscale` ModelPackage over that core (HEARTUpscalePackage).
// Weights: the four released checkpoints converted per-tensor exact by oracle/convert_weights.py and re-hosted under
// mlx-community/HEART-fp16 (shipping lane) and mlx-community/HEART-fp32 (parity lane) — `WeightSourcing`, one role per
// variant so a variant downloads only its own file. Parity: oracle/dump_goldens.py dumps per-sub-op goldens from the
// upstream PyTorch code on the CPU (S0–S7 in PORTING-SPEC.md); the author's ONNX exports are the second oracle.
// Licences: weights Apache-2.0 (Phips/HEART), port code Apache-2.0 (an isomorphic derivative of the upstream Apache-2.0
// arch files; see NOTICE).
let package = Package(
    name: "mlx-heart-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "HEARTMLX", targets: ["HEARTMLX"]),
        .library(name: "MLXHEART", targets: ["MLXHEART"]),
        .executable(name: "heart-smoke", targets: ["HEARTSmoke"]),   // CLI gate modes + drive the package
    ],
    dependencies: [
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.63.0"),
        .package(url: "https://github.com/xocialize/mlx-realesrgan-swift", from: "0.7.0"),  // MLXTileProcessor + PlaybackTier
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
    ],
    targets: [
        // Engine-agnostic core — NO MLXToolKit dep. Reuses the shipped tile driver rather than forking it.
        .target(
            name: "HEARTMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "RealESRGANMLX", package: "mlx-realesrgan-swift"),
            ]
        ),
        // MLXEngine `imageUpscale` wrapper over the local core.
        .target(
            name: "MLXHEART",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                "HEARTMLX",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            // The core's playback tier isn't Sendable-audited; the engine serializes lifecycle on
            // InferenceActor, so v5 mode keeps region-isolation a warning (same posture as the siblings).
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "HEARTMLXTests",
            dependencies: [
                "HEARTMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MLXHEARTTests",
            dependencies: [
                "MLXHEART",
                "HEARTMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformanceNN", package: "mlx-engine-swift"),   // the C14 INF walker
            ]
        ),
        // The CLI gate lane (S0–S7) + the real-engine smoke: keys · gate · run · engine · perf · tiles · cancel.
        .executableTarget(
            name: "HEARTSmoke",
            dependencies: [
                "HEARTMLX",
                "MLXHEART",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            path: "Sources/Smoke",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
