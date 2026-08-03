#if canImport(CoreML)
import Foundation
import Testing

@testable import MLXAudioSTT

@Suite("Nemotron CoreML Encoder Tests")
struct NemotronCoreMLEncoderTests {
    /// The offline Nemotron encoder reuses the generic fixed-shape Conformer CoreML encoder
    /// (`ConformerCoreMLEncoder`). A missing/invalid model must surface as a thrown error — the
    /// model then falls back to the MLX encoder — never a crash.
    @Test func conformerEncoderThrowsOnMissingModel() {
        let bogus = URL(fileURLWithPath: "/nonexistent/nemotron_enc.mlpackage")
        #expect(throws: (any Error).self) {
            _ = try ConformerCoreMLEncoder(
                modelURL: bogus, featIn: 128, fixedFrames: 1000, subsamplingFactor: 8)
        }
    }

    /// Nemotron subsampled-length math (8× dw-striding, pad-before-stride: `floor(L/2)+1`,
    /// log2(8)=3 times). This differs from Parakeet's `(L-1)/2+1` — e.g. 1000 → 126, not 125 —
    /// so the shared encoder is given Nemotron's per-stage step.
    @Test func subsampledLengthMatchesDwStriding() {
        let nemotronStep: (Int) -> Int = { $0 / 2 + 1 }
        #expect(ConformerCoreMLEncoder.subsampledLength(frames: 1000, subsamplingFactor: 8, step: nemotronStep) == 126)
        #expect(ConformerCoreMLEncoder.subsampledLength(frames: 112, subsamplingFactor: 8, step: nemotronStep) == 15)
        #expect(ConformerCoreMLEncoder.subsampledLength(frames: 1, subsamplingFactor: 8, step: nemotronStep) == 1)
    }
}
#endif
