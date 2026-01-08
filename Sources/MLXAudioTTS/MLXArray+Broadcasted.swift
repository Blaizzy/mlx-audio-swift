//
//  MLXArray+Broadcasted.swift
//  MLXAudioTTS
//

@preconcurrency import MLX

extension MLXArray {
    /// Convenience wrapper to match existing ported code.
    func broadcasted(to shape: [Int]) -> MLXArray {
        broadcast(self, to: shape)
    }
}
