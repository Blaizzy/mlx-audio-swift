// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "SimpleChat",
    platforms: [
        .iOS(.v26),
        .macOS(.v26)
    ],
    products: [
        .executable(name: "SimpleChat", targets: ["SimpleChat"])
    ],
    dependencies: [
        .package(path: "../.."),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", exact: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", exact: "0.8.1"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", exact: "1.1.9")
    ],
    targets: [
        .executableTarget(
            name: "SimpleChat",
            dependencies: [
                .product(name: "MLXAudioTTS", package: "mlx-audio-swift"),
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
                .product(name: "MLXAudioVAD", package: "mlx-audio-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ],
            path: "SimpleChat"
        )
    ]
)
