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
            name: "SileroVAD",
            targets: ["SileroVAD"]
        ),
        .library(
            name: "MLXAudioSTT",
            targets: ["MLXAudioSTT"]
        ),
        .executable(
            name: "stt-demo",
            targets: ["STTDemo"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.29.1"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.1.6"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.5.0"),
        .package(url: "https://github.com/vapor/console-kit.git", from: "4.15.0"),
    ],
    targets: [
        .target(
            name: "SileroVAD",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "MLXAudio/SileroVAD",
            resources: [
                .copy("Resources/silero_vad_16k.safetensors")
            ]
        ),
        .target(
            name: "MLXAudioSTT",
            dependencies: [
                "SileroVAD",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "MLXAudioSTT/Sources"
        ),
        .testTarget(
            name: "MLXAudioSTTTests",
            dependencies: [
                "MLXAudioSTT",
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "MLXAudioSTT/Tests",
            resources: [
                .copy("Resources")
            ]
        ),
        .executableTarget(
            name: "STTDemo",
            dependencies: [
                "MLXAudioSTT",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ConsoleKitTerminal", package: "console-kit"),
            ],
            path: "MLXAudioSTT/STTDemo"
        ),
    ]
)
