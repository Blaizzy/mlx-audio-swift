// swift-tools-version:6.2
import PackageDescription

let package = Package(
    name: "MLXAudio",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        // Core foundation library
        .library(name: "MLXAudioCore", targets: ["MLXAudioCore"]),

        // Audio codec implementations
        .library(name: "MLXAudioCodecs", targets: ["MLXAudioCodecs"]),

        // Text-to-Speech
        .library(name: "MLXAudioTTS", targets: ["MLXAudioTTS"]),

        // Speech-to-Text (placeholder)
        .library(name: "MLXAudioSTT", targets: ["MLXAudioSTT"]),

        // Speech-to-Speech
        .library(name: "MLXAudioSTS", targets: ["MLXAudioSTS"]),

        // SwiftUI components
        .library(name: "MLXAudioUI", targets: ["MLXAudioUI"]),

        // Legacy combined library (for backwards compatibility)
        .library(
            name: "MLXAudio",
            targets: ["MLXAudioCore", "MLXAudioCodecs", "MLXAudioTTS", "MLXAudioSTT", "MLXAudioSTS", "MLXAudioUI"]
        ),
        .executable(
            name: "mlx-audio-swift-tts",
            targets: ["mlx-audio-swift-tts"],
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMajor(from: "0.30.3")),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", .upToNextMajor(from: "2.30.3")),
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMajor(from: "1.1.6")),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", .upToNextMajor(from: "0.6.0"))
    ],
    targets: [
        // MARK: - MLXAudioCore
        .target(
            name: "MLXAudioCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/MLXAudioCore",
            swiftSettings: [
                .unsafeFlags(["-Xfrontend", "-warn-concurrency"], .when(configuration: .debug))
            ]
        ),

        // MARK: - MLXAudioCodecs
        .target(
            name: "MLXAudioCodecs",
            dependencies: [
                "MLXAudioCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/MLXAudioCodecs"
        ),

        // MARK: - MLXAudioTTS
        .target(
            name: "MLXAudioTTS",
            dependencies: [
                "MLXAudioCore",
                "MLXAudioCodecs",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/MLXAudioTTS"
        ),

        // MARK: - MLXAudioSTT
        .target(
            name: "MLXAudioSTT",
            dependencies: [
                "MLXAudioCore",
                "MLXAudioCodecs",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/MLXAudioSTT"
        ),

        // MARK: - MLXAudioSTS
        .target(
            name: "MLXAudioSTS",
            dependencies: [
                "MLXAudioCore",
                "MLXAudioTTS",
                "MLXAudioSTT",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/MLXAudioSTS"
        ),

        // MARK: - MLXAudioUI
        .target(
            name: "MLXAudioUI",
            dependencies: [
                "MLXAudioCore",
                "MLXAudioTTS",
                "MLXAudioSTS",
            ],
            path: "Sources/MLXAudioUI"
        ),
        
        .executableTarget(
            name: "mlx-audio-swift-tts",
            dependencies: ["MLXAudioCore", "MLXAudioTTS", "MLXAudioSTT"],
            path: "Sources/mlx-audio-swift-tts"
        ),

        // MARK: - Tests
        .testTarget(
            name: "MLXAudioTests",
            dependencies: [
                "MLXAudioCore",
                "MLXAudioCodecs",
                "MLXAudioTTS",
                "MLXAudioSTT",
                "MLXAudioSTS",
            ],
            path: "Tests",
            resources: [
                .copy("media")
            ]
        ),
    ]
)
