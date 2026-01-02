// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "VoicesApp",
    platforms: [
        .iOS(.v18),
        .macOS(.v15)
    ],
    products: [
        .executable(name: "VoicesApp", targets: ["VoicesApp"])
    ],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "VoicesApp",
            dependencies: [
                .product(name: "MLXAudioTTS", package: "mlx-audio-swift"),
                .product(name: "MLXAudioCore", package: "mlx-audio-swift")
            ],
            path: "VoicesApp"
        )
    ]
)
