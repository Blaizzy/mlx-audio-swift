import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Tokenizers

// MARK: - Audio Preprocessor

public class AudioPreprocessor {
    let config: PreprocessorConfig
    let melFilterbank: MLXArray

    public init(_ config: PreprocessorConfig) {
        self.config = config
        self.melFilterbank = melFilters(
            sampleRate: config.sampleRate,
            nFft: config.nFft,
            nMels: config.features,
            fMin: 0.0,
            fMax: Float(config.sampleRate / 2),
            norm: "slaney",
            melScale: .slaney
        )
    }

    public func callAsFunction(_ audio: MLXArray) -> MLXArray {
        let singleInput = audio.ndim == 1
        let input = singleInput ? audio.expandedDimensions(axis: 0) : audio
        let B = input.dim(0)
        var featuresList: [MLXArray] = []

        for i in 0..<B {
            var waveform = input[i]

            if config.dither > 0 {
                waveform = waveform + MLXArray(config.dither) * MLXRandom.normal(waveform.shape)
            }

            if config.preemph > 0 {
                let first = waveform[..<1]
                let rest = waveform[1...] - MLXArray(config.preemph) * waveform[..<(waveform.dim(0) - 1)]
                waveform = concatenated([first, rest])
            }

            let spec = stftConstantPad(
                waveform, nFft: config.nFft,
                hopLength: config.hopLength,
                winLength: config.winLength
            )

            let powerSpec = MLX.abs(spec).square()
            var melSpec = MLX.matmul(powerSpec, melFilterbank)

            if config.log {
                let logGuard: Float = 5.96e-8
                melSpec = MLX.log(melSpec + MLXArray(logGuard))
            }

            if config.normalize == "per_feature" {
                let validFrames = waveform.dim(0) / config.hopLength
                let n = min(validFrames, melSpec.dim(0))
                let validMel = melSpec[..<n]
                let mean = MLX.mean(validMel, axis: 0, keepDims: true)
                let variance = MLX.sum((validMel - mean).square(), axis: 0, keepDims: true) / MLXArray(Float(n - 1))
                let std = MLX.sqrt(variance) + MLXArray(Float(1e-5))
                melSpec = (melSpec - mean) / std
            }

            featuresList.append(melSpec)
        }

        let features = MLX.stacked(featuresList, axis: 0)
        return singleInput ? features.squeezed(axis: 0) : features
    }

    private func stftConstantPad(
        _ audio: MLXArray, nFft: Int, hopLength: Int, winLength: Int
    ) -> MLXArray {
        let padding = nFft / 2
        let prefix = MLXArray.zeros([padding])
        let suffix = MLXArray.zeros([padding])
        let padded = concatenated([prefix, audio, suffix])

        let paddedLen = padded.dim(0)
        let numFrames = 1 + (paddedLen - nFft) / hopLength

        let framesStacked = asStrided(padded, [numFrames, nFft], strides: [hopLength, 1], offset: 0)

        let window = hanningWindow(size: winLength)
        let effectiveWindow: MLXArray
        if winLength < nFft {
            let padLeft = (nFft - winLength) / 2
            let padRight = nFft - winLength - padLeft
            effectiveWindow = concatenated([MLXArray.zeros([padLeft]), window, MLXArray.zeros([padRight])])
        } else {
            effectiveWindow = window
        }

        let windowed = framesStacked * effectiveWindow
        return MLXFFT.rfft(windowed, axis: 1)
    }
}

// MARK: - Chat State

public class ChatState {
    public let processor: LFM2AudioProcessor
    public var textTokens: [Int]
    public var audioFeatures: MLXArray?
    public var audioOutCodes: [MLXArray]
    public var modalities: [LFMModality]
    public var currentTurn: String?

    public init(processor: LFM2AudioProcessor, addBos: Bool = true) {
        self.processor = processor
        self.textTokens = []
        self.audioFeatures = nil
        self.audioOutCodes = []
        self.modalities = []
        self.currentTurn = nil

        if addBos {
            textTokens.append(1) // BOS token
            modalities.append(.text)
        }
    }

    public func newTurn(role: String) {
        currentTurn = role
        let turnPrefix = "<|im_start|>\(role)\n"
        let tokens = processor.tokenize(turnPrefix)
        textTokens.append(contentsOf: tokens)
        for _ in tokens { modalities.append(.text) }
    }

    public func endTurn() {
        let tokens = processor.tokenize("<|im_end|>\n")
        textTokens.append(contentsOf: tokens)
        for _ in tokens { modalities.append(.text) }
        currentTurn = nil
    }

    public func addText(_ text: String) {
        let tokens = processor.tokenize(text)
        textTokens.append(contentsOf: tokens)
        for _ in tokens { modalities.append(.text) }
    }

    public func addAudioStartToken() {
        textTokens.append(lfmAudioStartToken)
        modalities.append(.text)
    }

    public func addAudio(_ audio: MLXArray, sampleRate: Int = 16000) {
        let features = processor.preprocessAudio(audio, sampleRate: sampleRate)
        if audioFeatures == nil {
            audioFeatures = features
        } else {
            audioFeatures = concatenated([audioFeatures!, features], axis: 0)
        }

        func convOutput(_ inputLen: Int, kernel: Int = 3, stride: Int = 2, padding: Int = 1) -> Int {
            (inputLen + 2 * padding - kernel) / stride + 1
        }
        let melFrames = features.dim(0)
        var t = convOutput(melFrames)
        t = convOutput(t)
        t = convOutput(t)

        for _ in 0..<t { modalities.append(.audioIn) }
    }

    public func append(token: MLXArray, modality: LFMModality) {
        if modality == .text {
            textTokens.append(token.item(Int.self))
        } else if modality == .audioOut {
            audioOutCodes.append(token)
        }
        modalities.append(modality)
    }

    public func getTextTokens() -> MLXArray {
        MLXArray(textTokens.map { Int32($0) }).expandedDimensions(axis: 0)
    }

    public func getAudioFeatures() -> MLXArray? {
        guard let af = audioFeatures else { return nil }
        return af.ndim == 2 ? af.expandedDimensions(axis: 0) : af
    }

    public func getModalities() -> MLXArray {
        MLXArray(modalities.map { Int32($0.rawValue) }).expandedDimensions(axis: 0)
    }
}

// MARK: - LFM2 Audio Processor

public class LFM2AudioProcessor {
    public let config: LFM2AudioConfig
    let audioPreprocessor: AudioPreprocessor
    private var _tokenizer: Tokenizer?
    public var modelPath: URL?

    public init(_ config: LFM2AudioConfig) {
        self.config = config
        self.audioPreprocessor = AudioPreprocessor(config.preprocessor)
    }

    public var tokenizer: Tokenizer {
        get throws {
            if let t = _tokenizer { return t }
            throw LFMAudioError.tokenizerNotLoaded
        }
    }

    public func loadTokenizer() async throws {
        guard let path = modelPath else {
            throw LFMAudioError.tokenizerNotLoaded
        }
        _tokenizer = try await AutoTokenizer.from(modelFolder: path)
    }

    public func tokenize(_ text: String) -> [Int] {
        do {
            let tok = try tokenizer
            return tok.encode(text: text, addSpecialTokens: false)
        } catch {
            return []
        }
    }

    public func preprocessAudio(_ audio: MLXArray, sampleRate: Int = 16000) -> MLXArray {
        audioPreprocessor(audio)
    }

    public func decodeText(_ tokens: [Int]) -> String {
        do {
            return try tokenizer.decode(tokens: tokens)
        } catch {
            return ""
        }
    }

    public static func fromPretrained(_ modelPath: URL, config: LFM2AudioConfig) async throws -> LFM2AudioProcessor {
        let processor = LFM2AudioProcessor(config)
        processor.modelPath = modelPath
        try await processor.loadTokenizer()
        return processor
    }
}

// MARK: - Errors

public enum LFMAudioError: Error, LocalizedError {
    case tokenizerNotLoaded
    case modelNotFound(String)
    case weightLoadingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .tokenizerNotLoaded: return "Tokenizer not loaded. Set modelPath first."
        case .modelNotFound(let path): return "Model not found at: \(path)"
        case .weightLoadingFailed(let msg): return "Weight loading failed: \(msg)"
        }
    }
}
