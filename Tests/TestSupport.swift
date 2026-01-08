//
//  TestSupport.swift
//  MLXAudioTests
//

import MLX

// Force CPU globally during test module initialization to avoid Metal library requirements.
private let _forceCPU: Void = {
    Device.setDefault(device: .cpu)
    return ()
}()

@discardableResult
func withCPU<T>(_ body: () async throws -> T) async rethrows -> T {
    // Ensure TaskLocal default is CPU for the test body.
    return try await Device.withDefaultDevice(.cpu) {
        try await body()
    }
}
