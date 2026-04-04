// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "ConsumerGraphFixture",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "ConsumerGraphFixture", targets: ["ConsumerGraphFixture"]),
    ],
    dependencies: [
        .package(path: "../.."),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", branch: "main"),
    ],
    targets: [
        .executableTarget(
            name: "ConsumerGraphFixture",
            dependencies: [
                .product(name: "MLXAudioCodecs", package: "mlx-audio-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
    ]
)
