import Foundation
import MLX
import MLXAudioCodecs

// MARK: - PCA State

struct EchoPCAState {
    let pcaComponents: MLXArray  // [inputDim, latentSize]
    let pcaMean: MLXArray        // [inputDim]
    let latentScale: Float
}

func echoLoadPCAState(from url: URL) throws -> EchoPCAState {
    let weights = try loadArrays(url: url)
    guard let components = weights["pca_components"],
          let mean = weights["pca_mean"] else {
        throw NSError(
            domain: "EchoTTS",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "PCA state missing pca_components or pca_mean"]
        )
    }

    let scale: Float
    if let scaleArray = weights["latent_scale"] {
        scale = scaleArray.item(Float.self)
    } else {
        scale = 1.0
    }

    return EchoPCAState(
        pcaComponents: components,
        pcaMean: mean,
        latentScale: scale
    )
}

// MARK: - Audio Encoding/Decoding with PCA

/// Encode audio through Fish S1 DAC + PCA to 80-dim latent.
/// Input: [B, 1, T]. Returns [B, T_ds, latentSize].
func echoAEEncode(fishAE: FishS1DAC, pcaState: EchoPCAState, audio: MLXArray) -> MLXArray {
    // Fish encode_zq: [B, 1, T] -> [B, 1024, T_ds]
    let zQ = fishAE.encodeZQ(audio)  // NCL

    // NCL -> NLC
    var z = zQ.transposed(0, 2, 1)  // [B, T_ds, 1024]

    // PCA transform: (z - mean) @ components.T * scale
    z = MLX.matmul(z - pcaState.pcaMean, pcaState.pcaComponents.transposed(0, 1))
    z = z * pcaState.latentScale

    return z  // [B, T_ds, latentSize=80]
}

/// Decode 80-dim latent through inverse PCA + Fish S1 DAC.
/// Input: [B, T_ds, latentSize]. Returns [B, 1, T].
func echoAEDecode(fishAE: FishS1DAC, pcaState: EchoPCAState, zQ: MLXArray) -> MLXArray {
    // Inverse PCA: (z / scale) @ components + mean
    var z = zQ / pcaState.latentScale
    z = MLX.matmul(z, pcaState.pcaComponents) + pcaState.pcaMean  // [B, T_ds, 1024]

    // NLC -> NCL
    z = z.transposed(0, 2, 1)  // [B, 1024, T_ds]

    // Fish decode_zq: [B, 1024, T_ds] -> [B, 1, T]
    return fishAE.decodeZQ(z)
}

// MARK: - Silence Detection

/// Find the point where generated latents flatten to zero (silence).
/// Returns the frame index, or nil if no flattening detected.
func echoFindFlatteningPoint(_ latents: MLXArray, windowSize: Int = 20, threshold: Float = 0.05) -> Int? {
    // latents: [B, T, latentSize] - use first batch
    let seqLen = latents.dim(1)
    guard seqLen > windowSize else { return nil }

    for i in 0..<(seqLen - windowSize) {
        let window = latents[0, i..<(i + windowSize), 0...]
        let stdVal = MLX.std(window).item(Float.self)
        let meanVal = MLX.abs(MLX.mean(window)).item(Float.self)
        if stdVal < threshold && meanVal < threshold {
            return i
        }
    }
    return nil
}

/// Crop audio at the flattening point.
func echoCropAudioToFlatteningPoint(
    audio: MLXArray, flatteningPoint: Int?, downsampleFactor: Int
) -> MLXArray {
    guard let point = flatteningPoint else { return audio }
    let samplePoint = point * downsampleFactor
    let audioLength = audio.dim(-1)
    if samplePoint > 0 && samplePoint < audioLength {
        return audio[0..., 0..., ..<samplePoint]
    }
    return audio
}

// MARK: - Speaker Latent Extraction

/// Extract speaker latent from reference audio.
/// Input: reference audio [1, 1, T].
/// Returns (speakerLatent [1, T_latent, latentSize], mask [1, T_latent]).
func echoGetSpeakerLatentAndMask(
    fishAE: FishS1DAC, pcaState: EchoPCAState, audio: MLXArray, config: EchoTTSConfig
) -> (MLXArray, MLXArray) {
    let maxSamples = config.maxSpeakerLatentLength * config.audioDownsampleFactor
    let chunkSamples = config.sampler.sequenceLength * config.audioDownsampleFactor

    // Limit audio length
    var refAudio = audio
    if refAudio.dim(-1) > maxSamples {
        refAudio = refAudio[0..., 0..., ..<maxSamples]
    }

    let totalSamples = refAudio.dim(-1)
    var allLatents: [MLXArray] = []

    // Process in chunks
    var start = 0
    while start < totalSamples {
        let end = min(start + chunkSamples, totalSamples)
        let chunk = refAudio[0..., 0..., start..<end]

        // Pad chunk if needed
        var paddedChunk = chunk
        if chunk.dim(-1) < chunkSamples {
            let padAmount = chunkSamples - chunk.dim(-1)
            paddedChunk = MLX.padded(chunk, widths: [.init(0), .init(0), .init((0, padAmount))])
        }

        let latent = echoAEEncode(fishAE: fishAE, pcaState: pcaState, audio: paddedChunk)
        allLatents.append(latent)
        start = end
    }

    // Concatenate all latent chunks
    var speakerLatent = MLX.concatenated(allLatents, axis: 1)  // [1, totalFrames, latentSize]

    // Trim to actual length (remove padding from last chunk)
    let actualFrames = totalSamples / config.audioDownsampleFactor
    if speakerLatent.dim(1) > actualFrames {
        speakerLatent = speakerLatent[0..., ..<actualFrames, 0...]
    }

    // Create mask
    let latentLen = speakerLatent.dim(1)
    let mask = MLXArray.ones([1, latentLen])

    // Ensure divisible by speaker_patch_size
    let patchSize = config.dit.speakerPatchSize
    let remainder = latentLen % patchSize
    if remainder != 0 {
        let padFrames = patchSize - remainder
        speakerLatent = MLX.padded(speakerLatent, widths: [.init(0), .init((0, padFrames)), .init(0)])
        let maskPad = MLXArray.zeros([1, padFrames])
        let newMask = MLX.concatenated([mask, maskPad], axis: 1)
        return (speakerLatent, newMask)
    }

    return (speakerLatent, mask)
}
