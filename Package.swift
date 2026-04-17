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

        // Voice Activity Detection / Speaker Diarization
        .library(name: "MLXAudioVAD", targets: ["MLXAudioVAD"]),

        // Language Identification
        .library(name: "MLXAudioLID", targets: ["MLXAudioLID"]),

        // Speech-to-Speech
        .library(name: "MLXAudioSTS", targets: ["MLXAudioSTS"]),

        // SwiftUI components
        .library(name: "MLXAudioUI", targets: ["MLXAudioUI"]),

        // Grapheme-to-Phoneme (neural ByT5 + dictionary lexicons)
        .library(name: "MLXAudioG2P", targets: ["MLXAudioG2P"]),

        // Legacy combined library (for backwards compatibility)
        .library(
            name: "MLXAudio",
            targets: ["MLXAudioCore", "MLXAudioCodecs", "MLXAudioTTS", "MLXAudioSTT", "MLXAudioVAD", "MLXAudioLID", "MLXAudioSTS", "MLXAudioUI", "MLXAudioG2P"]
        ),
        .executable(
            name: "mlx-audio-swift-tts",
            targets: ["mlx-audio-swift-tts"],
        ),
        .executable(
            name: "mlx-audio-swift-codec",
            targets: ["mlx-audio-swift-codec"],
        ),
        .executable(
            name: "mlx-audio-swift-sts",
            targets: ["mlx-audio-swift-sts"],
        ),
        .executable(
            name: "mlx-audio-swift-stt",
            targets: ["mlx-audio-swift-stt"],
        ),
        .executable(
            name: "mlx-audio-swift-lid",
            targets: ["mlx-audio-swift-lid"],
        ),

    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMajor(from: "0.30.6")),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", .upToNextMajor(from: "3.31.3")),
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMajor(from: "1.1.6")),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", .upToNextMajor(from: "0.8.1"))
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
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/MLXAudioCodecs"
        ),

        // MARK: - MLXAudioTTS
        .target(
            name: "MLXAudioTTS",
            dependencies: [
                "MLXAudioCore",
                "MLXAudioCodecs",
                "MLXAudioG2P",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/MLXAudioTTS",
            exclude: [
                "Models/Chatterbox/README.md",
                "Models/EchoTTS/README.md",
                "Models/FishSpeech/README.md",
                "Models/Llama/README.md",
                "Models/Marvis/README.md",
                "Models/PocketTTS/README.md",
                "Models/Qwen3/README.md",
                "Models/Qwen3TTS/README.md",
                "Models/Soprano/README.md",
                "Models/StyleTTS2/KittenTTS/README.md",
                "Models/StyleTTS2/Kokoro/README.md",
            ]
        ),

        // MARK: - MLXAudioSTT
        .target(
            name: "MLXAudioSTT",
            dependencies: [
                "MLXAudioCore",
                "MLXAudioCodecs",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/MLXAudioSTT",
            exclude: [
                "Models/CohereTranscribe/README.md",
                "Models/FireRedASR2/README.md",
                "Models/GLMASR/README.md",
                "Models/GraniteSpeech/README.md",
                "Models/Parakeet/README.md",
                "Models/Qwen3ASR/README.md",
                "Models/SenseVoice/README.md",
                "Models/VoxtralRealtime/README.md",
            ]
        ),

        // MARK: - MLXAudioVAD
        .target(
            name: "MLXAudioVAD",
            dependencies: [
                "MLXAudioCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/MLXAudioVAD",
            exclude: [
                "Models/SmartTurn/README.md",
                "Models/Sortformer/README.md",
            ]
        ),

        // MARK: - MLXAudioLID
        .target(
            name: "MLXAudioLID",
            dependencies: [
                "MLXAudioCore",
                "MLXAudioCodecs",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/MLXAudioLID",
            exclude: [
                "README.md",
            ]
        ),

        // MARK: - MLXAudioSTS
        .target(
            name: "MLXAudioSTS",
            dependencies: [
                "MLXAudioCore",
                "MLXAudioCodecs",
                "MLXAudioTTS",
                "MLXAudioSTT",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/MLXAudioSTS",
            exclude: [
                "Models/LFMAudio/README.md",
                "Models/SAMAudio/README.md",
            ]
        ),

        // MARK: - MLXAudioG2P
        .target(
            name: "MLXAudioG2P",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/MLXAudioG2P"
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
            path: "Sources/Tools/mlx-audio-swift-tts",
            exclude: [
                "README.md",
            ]
        ),
        .executableTarget(
            name: "mlx-audio-swift-codec",
            dependencies: ["MLXAudioCore", "MLXAudioCodecs"],
            path: "Sources/Tools/mlx-audio-swift-codec",
            exclude: [
                "README.md",
            ]
        ),
        .executableTarget(
            name: "mlx-audio-swift-sts",
            dependencies: ["MLXAudioCore", "MLXAudioSTS"],
            path: "Sources/Tools/mlx-audio-swift-sts",
            exclude: [
                "README.md",
            ]
        ),
        .executableTarget(
            name: "mlx-audio-swift-stt",
            dependencies: ["MLXAudioCore", "MLXAudioSTT"],
            path: "Sources/Tools/mlx-audio-swift-stt",
            exclude: [
                "README.md",
            ]
        ),
        .executableTarget(
            name: "mlx-audio-swift-lid",
            dependencies: ["MLXAudioCore", "MLXAudioLID"],
            path: "Sources/Tools/mlx-audio-swift-lid",
            exclude: [
                "README.md",
            ]
        ),

        // MARK: - Tests
        .testTarget(
            name: "MLXAudioTests",
            dependencies: [
                "MLXAudioCore",
                "MLXAudioCodecs",
                "MLXAudioTTS",
                "MLXAudioSTT",
                "MLXAudioVAD",
                "MLXAudioSTS",
                "MLXAudioLID",
                "mlx-audio-swift-lid",
                "MLXAudioG2P",
            ],
            path: "Tests",
            resources: [
                .copy("media")
            ]
        ),
    ]
)
