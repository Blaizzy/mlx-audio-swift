// swift-tools-version:5.9
import PackageDescription

// NOTE: TTS targets are temporarily disabled due to path issues.
// The ESpeakNG.xcframework is located at MLXAudio/Kokoro/Frameworks/ but Package.swift
// expects it at mlx_audio_swift/tts/MLXAudio/Kokoro/Frameworks/.
// This will be resolved when the TTS module structure is updated.

let package = Package(
    name: "mlx-audio",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(
            name: "MLXAudioSTT",
            targets: ["MLXAudioSTT"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.29.0"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "0.1.18"),
    ],
    targets: [
        .target(
            name: "MLXAudioSTT",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "mlx_audio_swift/stt/MLXAudioSTT"
        ),
        .testTarget(
            name: "MLXAudioSTTTests",
            dependencies: [
                "MLXAudioSTT",
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "mlx_audio_swift/stt/Tests"
        ),
    ]
)
