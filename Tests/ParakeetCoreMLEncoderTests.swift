#if canImport(CoreML)
import Foundation
import Testing

@testable import MLXAudioSTT

@Suite("Parakeet CoreML Encoder Tests")
struct ParakeetCoreMLEncoderTests {
    /// The wrapper's output-length math must match `ParakeetModel.computeEncodedLengths`
    /// (NeMo dw-striding: `floor((L-1)/2)+1`, log2(factor) times).
    @Test func subsampledLengthMatchesDwStriding() {
        #expect(ParakeetCoreMLEncoder.subsampledLength(frames: 1000, subsamplingFactor: 8) == 125)
        #expect(ParakeetCoreMLEncoder.subsampledLength(frames: 995, subsamplingFactor: 8) == 125)
        #expect(ParakeetCoreMLEncoder.subsampledLength(frames: 128, subsamplingFactor: 8) == 16)
        #expect(ParakeetCoreMLEncoder.subsampledLength(frames: 1, subsamplingFactor: 8) == 1)
    }

    /// A missing/invalid `.mlpackage` must surface as a thrown error (the model then falls
    /// back to the MLX encoder), never a crash.
    @Test func throwsOnMissingModel() {
        let bogus = URL(fileURLWithPath: "/nonexistent/parakeet_enc.mlpackage")
        #expect(throws: (any Error).self) {
            _ = try ParakeetCoreMLEncoder(
                modelURL: bogus, featIn: 128, fixedFrames: 1000, subsamplingFactor: 8)
        }
    }

    /// Resolving the downloaded encoder picks the `.mlpackage` (or `.mlmodelc`) directory.
    @Test func findsEncoderPackageInDirectory() throws {
        let fm = FileManager.default
        let base = fm.temporaryDirectory.appendingPathComponent("parakeet-coreml-findtest")
        try? fm.removeItem(at: base)
        let pkg = base.appendingPathComponent("enc.mlpackage")
        try fm.createDirectory(at: pkg, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: base) }

        #expect(ParakeetModel.findEncoderPackage(in: base)?.lastPathComponent == "enc.mlpackage")
        #expect(ParakeetModel.findEncoderPackage(in: pkg) == nil)  // empty dir → nothing
    }
}
#endif
