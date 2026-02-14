// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "VoxtralApp",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "VoxtralApp", targets: ["VoxtralApp"])
    ],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "VoxtralApp",
            dependencies: [
                .product(name: "MLXAudioSTT", package: "mlx-audio-swift"),
                .product(name: "MLXAudioCore", package: "mlx-audio-swift")
            ],
            path: "VoxtralApp",
            exclude: ["Info.plist"]
        )
    ]
)
