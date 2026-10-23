import MLX

struct VoxtralAudio {
    let config: VoxtralAudioConfig
    var samplesPerChunk: Int { config.maxSourcePositions * 2 * 160 }
    var tokensPerChunk: Int {
        config.maxSourcePositions * config.hiddenSize / config.intermediateSize
    }

    func callAsFunction(_ audio: MLXArray) -> MLXArray {
        precondition(
            (audio.ndim == 1 || audio.ndim == 2) && audio.size > 0,
            "Expected nonempty mono or time-major audio at 16 kHz")
        let mono = (audio.ndim == 2 ? audio.mean(axis: -1) : audio).asType(.float32)
        let chunks = (mono.size + samplesPerChunk - 1) / samplesPerChunk
        let padded = MLX.padded(mono, widths: [.init((0, chunks * samplesPerChunk - mono.size))])
        // Normalize the entire padded recording before splitting its mel frames.
        let mel = WhisperAudio.logMelSpectrogram(padded, nMels: config.numMelBins)
        return mel.reshaped(config.numMelBins, chunks, config.maxSourcePositions * 2).transposed(
            1, 2, 0)
    }
}
