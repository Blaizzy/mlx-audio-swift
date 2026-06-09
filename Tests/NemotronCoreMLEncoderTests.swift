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

    /// Subsampled-length math (8× dw-striding: `floor((L-1)/2)+1`, log2(8)=3 times). Nemotron's
    /// FastConformer uses the same subsampling as Parakeet, so the shared encoder applies.
    @Test func subsampledLengthMatchesDwStriding() {
        #expect(ConformerCoreMLEncoder.subsampledLength(frames: 1000, subsamplingFactor: 8) == 125)
        #expect(ConformerCoreMLEncoder.subsampledLength(frames: 112, subsamplingFactor: 8) == 14)
        #expect(ConformerCoreMLEncoder.subsampledLength(frames: 1, subsamplingFactor: 8) == 1)
    }
}
#endif
