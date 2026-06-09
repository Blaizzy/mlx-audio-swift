#if canImport(CoreML)
import CoreML
import Foundation

/// Nemotron's FastConformer encoder has the same fixed-shape I/O as Parakeet's Conformer
/// (`[1, T, featIn]` mel in, `[1, T', dModel]` + lengths out, 8× dw-striding subsampling), so
/// the offline CoreML/ANE encoder is the *same* generic implementation — reused, not duplicated.
public typealias ConformerCoreMLEncoder = ParakeetCoreMLEncoder

public extension NemotronASRModel {
    /// Enable the **offline** CoreML/ANE encoder from a local `.mlpackage` / `.mlmodelc`.
    ///
    /// `fixedFrames` must match the converted model (default 1000 = 10 s of 10 ms-hop mel). Each
    /// `decode()` chunk's mel is padded/cropped to this length, so `generate()` auto-clamps its
    /// `chunkDuration` to ≤ that length and its overlap-merge stitches arbitrary-length audio.
    ///
    /// Streaming (`generateStream`) is unaffected and still uses the MLX cache-aware path; a
    /// CoreML streaming encoder is a separate addition.
    func enableCoreMLEncoder(modelURL: URL, fixedFrames: Int = 1000) throws {
        coreMLEncoder = try ConformerCoreMLEncoder(
            modelURL: modelURL,
            featIn: encoderConfig.featIn,
            fixedFrames: fixedFrames,
            subsamplingFactor: encoderConfig.subsamplingFactor
        )
    }
}
#endif
