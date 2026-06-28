#if canImport(CoreML)
import CoreML
import Foundation
import HuggingFace

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
            subsamplingFactor: encoderConfig.subsamplingFactor,
            dModel: encoderConfig.dModel,
            // Nemotron's dw-striding subsampling pads before each stride (pad 2/1, kernel 3,
            // stride 2) → `floor(L/2)+1` per stage, unlike Parakeet's `(L-1)/2+1`. Without this
            // the ANE output is cropped one valid frame short per chunk.
            subsampledLengthStep: { $0 / 2 + 1 }
        )
    }

    /// Hugging Face repo with the prebuilt CoreML/ANE encoder `.mlpackage` (matched to the MLX
    /// weights of `nemotron-3.5-asr-streaming-0.6b`).
    static var defaultANEEncoderRepo: String { "beshkenadze/nemotron-3.5-asr-streaming-0.6b-coreml-ane" }

    /// Download the CoreML encoder `.mlpackage` from a Hugging Face repo, then route the offline
    /// `decode()` path through it. Reuses Parakeet's downloader (the encoder package is generic).
    func enableCoreMLEncoder(repo: String, cache: HubCache = .default) async throws {
        let url = try await ParakeetModel.downloadANEEncoderPackage(repo: repo, cache: cache)
        try enableCoreMLEncoder(modelURL: url)
    }

    /// Enable the cache-aware **streaming** CoreML/ANE encoder from a local `.mlpackage`.
    ///
    /// `generateStream` then runs the conformer encoder on the ANE via the validated uniform-F
    /// feeding (`[preFrames prev-mel ++ newFrames new-mel]`, stride `newFrames`) with manual cache
    /// threading; the prompt MLP and RNN-T decode stay in MLX. `preFrames`/`newFrames` come from the
    /// model's NeMo `streaming_cfg` (`pre_encode_cache_size[1]` / `chunk_size[1]`) and must match the
    /// converted model. The attention cache is the encoder's left context and the conv cache is
    /// `convKernelSize - 1`, both derived from the loaded config.
    func enableCoreMLStreamingEncoder(
        modelURL: URL,
        preFrames: Int = 9,
        newFrames: Int = 112
    ) throws {
        streamingCoreMLEncoder = try NemotronCoreMLStreamingEncoder(
            modelURL: modelURL,
            featIn: encoderConfig.featIn,
            dModel: encoderConfig.dModel,
            subsamplingFactor: encoderConfig.subsamplingFactor,
            preFrames: preFrames,
            newFrames: newFrames,
            layers: encoderConfig.nLayers,
            attnCache: defaultAttContextSize.first ?? 70,
            convCache: encoderConfig.convKernelSize - 1
        )
    }

    /// Hugging Face repo with the prebuilt cache-aware **streaming** CoreML/ANE encoder
    /// `.mlpackage` (matched to the MLX weights of `nemotron-3.5-asr-streaming-0.6b`).
    static var defaultANEStreamingEncoderRepo: String { "beshkenadze/nemotron-3.5-asr-streaming-0.6b-coreml-ane-stream" }

    /// Download the streaming CoreML encoder `.mlpackage` from a Hugging Face repo and route the
    /// `generateStream` path through it. Reuses Parakeet's downloader (the package is the same shape).
    func enableCoreMLStreamingEncoder(repo: String, cache: HubCache = .default) async throws {
        let url = try await ParakeetModel.downloadANEEncoderPackage(repo: repo, cache: cache)
        try enableCoreMLStreamingEncoder(modelURL: url)
    }
}
#endif
