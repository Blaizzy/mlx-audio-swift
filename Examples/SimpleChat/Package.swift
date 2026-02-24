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
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "SimpleChat",
            dependencies: [
                .product(name: "MLXAudioTTS", package: "mlx-audio-swift"),
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
                .product(name: "MLXAudioVAD", package: "mlx-audio-swift")
            ],
            path: "SimpleChat"
        )
    ]
)
