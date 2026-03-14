import Foundation
import MLX
import MLXAudioCodecs
import MLXAudioCore

protocol EchoTTSAudioCodec: AnyObject {
    func encodeZQ(_ audioData: MLXArray) -> MLXArray
    func decodeZQ(_ zQ: MLXArray) -> MLXArray
}

extension FishS1DAC: EchoTTSAudioCodec {}

public struct EchoTTSPCAState {
    public let pcaComponents: MLXArray
    public let pcaMean: MLXArray
    public let latentScale: Float

    public init(pcaComponents: MLXArray, pcaMean: MLXArray, latentScale: Float) {
        self.pcaComponents = pcaComponents
        self.pcaMean = pcaMean
        self.latentScale = latentScale
    }
}

func loadEchoTTSPCAState(from url: URL) throws -> EchoTTSPCAState {
    let tensors = try loadArrays(url: url)
    guard
        let pcaComponents = tensors["pca_components"],
        let pcaMean = tensors["pca_mean"],
        let latentScale = tensors["latent_scale"]
    else {
        throw AudioGenerationError.modelNotInitialized("Missing PCA tensors at \(url.path)")
    }

    return EchoTTSPCAState(
        pcaComponents: pcaComponents,
        pcaMean: pcaMean,
        latentScale: latentScale.item(Float.self)
    )
}

func echoTtsAEEncode(
    codec: EchoTTSAudioCodec,
    pcaState: EchoTTSPCAState,
    audio: MLXArray
) -> MLXArray {
    let zQ = codec.encodeZQ(audio).asType(.float32).transposed(0, 2, 1)
    let centered = zQ - pcaState.pcaMean
    let projected = MLX.matmul(centered, pcaState.pcaComponents.transposed(1, 0))
    return projected * pcaState.latentScale
}

func echoTtsAEDecode(
    codec: EchoTTSAudioCodec,
    pcaState: EchoTTSPCAState,
    latent: MLXArray
) -> MLXArray {
    let unscaled = latent / pcaState.latentScale
    let restored = MLX.matmul(unscaled, pcaState.pcaComponents) + pcaState.pcaMean
    return codec.decodeZQ(restored.transposed(0, 2, 1).asType(.float32)).asType(.float32)
}

func echoTtsFindFlatteningPoint(
    _ data: MLXArray,
    targetValue: Float = 0,
    windowSize: Int = 20,
    stdThreshold: Float = 0.05
) -> Int {
    let padded = MLX.concatenated(
        [data, MLXArray.zeros([windowSize, data.shape[1]], dtype: data.dtype)],
        axis: 0
    )
    let maxStart = max(padded.shape[0] - windowSize, 0)
    for start in 0 ..< maxStart {
        let window = padded[start..<(start + windowSize), 0...]
        let std = MLX.std(window).item(Float.self)
        let mean = MLX.mean(window).item(Float.self)
        if std < stdThreshold && abs(mean - targetValue) < 0.1 {
            return start
        }
    }
    return data.shape[0]
}

func echoTtsCropAudioToFlatteningPoint(
    audio: MLXArray,
    latent: MLXArray,
    audioDownsampleFactor: Int = 2_048
) -> MLXArray {
    let flatteningPoint = echoTtsFindFlatteningPoint(latent)
    return audio[0..., 0..., 0..<(flatteningPoint * audioDownsampleFactor)]
}

func echoTtsGetSpeakerLatentAndMask(
    codec: EchoTTSAudioCodec,
    pcaState: EchoTTSPCAState,
    audio: MLXArray,
    maxSpeakerLatentLength: Int = 6_400,
    audioChunkSize: Int = 640 * 2_048,
    audioDownsampleFactor: Int = 2_048,
    padToMax: Bool = false,
    divisByPatchSize: Int? = 4
) -> (speakerLatent: MLXArray, speakerMask: MLXArray) {
    let maxAudioLength = maxSpeakerLatentLength * audioDownsampleFactor
    let clippedLength = min(audio.shape[1], maxAudioLength)
    let clippedAudio = audio[0..., 0..<clippedLength]

    var latentChunks: [MLXArray] = []
    if clippedLength > 0 {
        for start in stride(from: 0, to: clippedLength, by: audioChunkSize) {
            let end = min(clippedLength, start + audioChunkSize)
            var chunk = clippedAudio[0..., start..<end]
            if chunk.shape[1] < audioChunkSize {
                let pad = audioChunkSize - chunk.shape[1]
                chunk = MLX.padded(chunk, widths: [IntOrPair(0), IntOrPair((0, pad))])
            }
            let encoded = echoTtsAEEncode(
                codec: codec,
                pcaState: pcaState,
                audio: chunk.expandedDimensions(axis: 1)
            )
            latentChunks.append(encoded)
        }
    }

    let latentDim = pcaState.pcaComponents.shape[0]
    var speakerLatent = latentChunks.isEmpty
        ? MLXArray.zeros([1, 0, latentDim], dtype: .float32)
        : MLX.concatenated(latentChunks, axis: 1)

    let actualLatentLength = clippedLength / audioDownsampleFactor
    var speakerMask = MLX.broadcast(
        MLX.arange(speakerLatent.shape[1], dtype: .int32).expandedDimensions(axis: 0),
        to: [1, speakerLatent.shape[1]]
    ) .< actualLatentLength

    if padToMax && speakerLatent.shape[1] < maxSpeakerLatentLength {
        let pad = maxSpeakerLatentLength - speakerLatent.shape[1]
        speakerLatent = MLX.padded(
            speakerLatent,
            widths: [IntOrPair(0), IntOrPair((0, pad)), IntOrPair(0)]
        )
        speakerMask = MLX.padded(
            speakerMask,
            widths: [IntOrPair(0), IntOrPair((0, pad))]
        )
    } else if !padToMax {
        speakerLatent = speakerLatent[0..., 0..<actualLatentLength, 0...]
        speakerMask = speakerMask[0..., 0..<actualLatentLength]
    }

    if let divisByPatchSize, speakerLatent.shape[1] > 0 {
        let limit = (speakerLatent.shape[1] / divisByPatchSize) * divisByPatchSize
        speakerLatent = speakerLatent[0..., 0..<limit, 0...]
        speakerMask = speakerMask[0..., 0..<limit]
    }

    return (speakerLatent, speakerMask)
}
