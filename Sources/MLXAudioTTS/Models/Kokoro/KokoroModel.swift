import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXNN
@preconcurrency import MLXLMCommon

/// Kokoro TTS model — 82M parameter text-to-speech.
///
/// Kokoro generates natural-sounding speech from phonemized text using:
/// 1. BERT-based phoneme encoding (PLBERT/Albert)
/// 2. Duration prediction with BiLSTM
/// 3. Prosody prediction (F0 pitch and voicing)
/// 4. iSTFT HiFi-GAN decoder for waveform synthesis
///
/// By default, Kokoro expects pre-phonemized IPA text. To use plain text input,
/// provide a ``TextProcessor`` implementation (e.g., MisakiSwift for English G2P):
///
/// ```swift
/// // With pre-phonemized text (default)
/// let model = try await KokoroModel.fromPretrained("mlx-community/Kokoro-82M-bf16")
/// let audio = try await model.generate(text: "hɛloʊ wˈɜɹld", voice: "af_heart")
///
/// // With a TextProcessor for plain text
/// let model = try await KokoroModel.fromPretrained(
///     "mlx-community/Kokoro-82M-bf16",
///     textProcessor: myG2PProcessor
/// )
/// let audio = try await model.generate(text: "Hello world", voice: "af_heart")
/// ```
public final class KokoroModel: SpeechGenerationModel, @unchecked Sendable {

    public enum KokoroError: Error {
        case tooManyTokens
        case configNotFound
        case weightsNotFound
        case voiceNotFound(String)
    }

    /// Maximum number of input tokens.
    public static let maxTokenCount = 510

    public let sampleRate: Int = 24000

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters()
    }

    let config: KokoroConfig
    private let bert: KokoroBERT
    private let bertEncoder: Linear
    private let durationEncoder: KokoroDurationEncoder
    private let predictorLSTM: KokoroBiLSTM
    private let durationProj: Linear
    private let prosodyPredictor: KokoroProsodyPredictor
    private let textEncoder: KokoroTextEncoder
    private let decoder: KokoroAudioDecoder

    /// Directory containing the downloaded model (for loading voices).
    private let modelDirectory: URL?

    /// Optional text processor for converting plain text to phonemes.
    /// When nil, input text is expected to be pre-phonemized IPA.
    private var textProcessor: TextProcessor?

    /// Cached voice embeddings.
    private var voiceCache: [String: MLXArray] = [:]

    /// Speed multiplier for speech rate (0.5–2.0). Higher = faster.
    /// Set from the host app to override the default 1.0.
    public var speed: Float = 1.0

    init(
        config: KokoroConfig, weights: [String: MLXArray],
        modelDirectory: URL? = nil, textProcessor: TextProcessor? = nil
    ) {
        self.config = config
        self.modelDirectory = modelDirectory
        self.textProcessor = textProcessor

        let albertArgs = KokoroAlbertArgs(
            numHiddenLayers: config.plbert.numHiddenLayers,
            numAttentionHeads: config.plbert.numAttentionHeads,
            hiddenSize: config.plbert.hiddenSize,
            intermediateSize: config.plbert.intermediateSize,
            vocabSize: config.nToken
        )

        bert = KokoroBERT(weights: weights, config: albertArgs)
        bertEncoder = Linear(
            weight: weights["bert_encoder.weight"]!,
            bias: weights["bert_encoder.bias"]!
        )

        durationEncoder = KokoroDurationEncoder(
            weights: weights,
            dModel: config.hiddenDim,
            styDim: config.styleDim,
            nlayers: config.nLayer
        )

        predictorLSTM = KokoroBiLSTM(
            inputSize: config.hiddenDim + config.styleDim,
            hiddenSize: config.hiddenDim / 2,
            wxForward: weights["predictor.lstm.weight_ih_l0"]!,
            whForward: weights["predictor.lstm.weight_hh_l0"]!,
            biasIhForward: weights["predictor.lstm.bias_ih_l0"]!,
            biasHhForward: weights["predictor.lstm.bias_hh_l0"]!,
            wxBackward: weights["predictor.lstm.weight_ih_l0_reverse"]!,
            whBackward: weights["predictor.lstm.weight_hh_l0_reverse"]!,
            biasIhBackward: weights["predictor.lstm.bias_ih_l0_reverse"]!,
            biasHhBackward: weights["predictor.lstm.bias_hh_l0_reverse"]!
        )

        durationProj = Linear(
            weight: weights["predictor.duration_proj.linear_layer.weight"]!,
            bias: weights["predictor.duration_proj.linear_layer.bias"]!
        )

        prosodyPredictor = KokoroProsodyPredictor(
            weights: weights, styleDim: config.styleDim, dHid: config.hiddenDim
        )

        textEncoder = KokoroTextEncoder(
            weights: weights, channels: config.hiddenDim,
            kernelSize: config.textEncoderKernelSize, depth: config.nLayer
        )

        decoder = KokoroAudioDecoder(weights: weights, config: config)
    }

    // MARK: - SpeechGenerationModel

    public func generate(
        text: String, voice: String?, refAudio: MLXArray?,
        refText: String?, language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        let voiceEmbedding: MLXArray
        if let refAudio {
            voiceEmbedding = refAudio
        } else {
            let voiceName = voice ?? "af_heart"
            voiceEmbedding = try loadVoice(named: voiceName)
        }

        // If a text processor is set, convert plain text to phonemes first.
        // Otherwise, assume the input is already phonemized IPA.
        let phonemizedText: String
        if let textProcessor {
            phonemizedText = try textProcessor.process(text: text, language: language)
        } else {
            phonemizedText = text
        }

        return try synthesize(phonemizedText: phonemizedText, voice: voiceEmbedding, speed: speed)
    }

    public func generateStream(
        text: String, voice: String?, refAudio: MLXArray?,
        refText: String?, language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()

        Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish(throwing: AudioGenerationError.modelNotInitialized("Model deallocated"))
                return
            }
            do {
                let audio = try await self.generate(
                    text: text, voice: voice, refAudio: refAudio,
                    refText: refText, language: language,
                    generationParameters: generationParameters
                )
                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }

        return stream
    }

    // MARK: - Core Synthesis

    /// Synthesize audio from phonemized text.
    public func synthesize(phonemizedText: String, voice: MLXArray, speed: Float = 1.0) throws -> MLXArray {
        // Step 1: Tokenize
        let inputIds = KokoroTokenizer.tokenize(phonemizedText: phonemizedText, vocab: config.vocab)
        guard inputIds.count <= Self.maxTokenCount else {
            throw KokoroError.tooManyTokens
        }

        // Step 2: Prepare input tensors (pad with 0 tokens)
        let paddedArray = [0] + inputIds + [0]
        let paddedInputIds = MLXArray(paddedArray).expandedDimensions(axes: [0])
        let inputLengths = MLXArray(paddedInputIds.dim(-1))
        let inputLengthMax: Int = inputLengths.max().item()

        var textMask = MLXArray(0..<inputLengthMax)
        textMask = textMask + 1 .> inputLengths
        textMask = textMask.expandedDimensions(axes: [0])

        let maskBools: [Bool] = textMask.asArray(Bool.self)
        let attentionMask = MLXArray(maskBools.map { !$0 ? 1 : 0 }).reshaped(textMask.shape)

        // Step 3: Extract style embeddings from voice
        let referenceStyle = voice[inputIds.count - 1, 0...1, 0...]
        let globalStyle = referenceStyle[0...1, 128...]
        let acousticStyle = referenceStyle[0...1, 0...127]

        // Step 4: BERT encode + duration prediction
        let (bertOutput, _) = bert(paddedInputIds, attentionMask: attentionMask)
        let bertEncoded = bertEncoder(bertOutput).transposed(0, 2, 1)
        let durationFeatures = durationEncoder(
            bertEncoded, style: globalStyle,
            textLengths: inputLengths, m: textMask
        )

        // Step 5: Predict durations
        let (lstmOutput, _) = predictorLSTM(durationFeatures)
        let durationLogits = durationProj(lstmOutput)
        let durationSigmoid = MLX.sigmoid(durationLogits).sum(axis: -1) / speed
        let predictedDurations = MLX.clip(durationSigmoid.round(), min: 1).asType(.int32)[0]

        // Step 6: Create alignment target
        let alignmentTarget = createAlignmentTarget(
            durations: predictedDurations, batchSize: paddedInputIds.shape[1]
        )

        // Step 7: Aligned encoding + prosody
        let alignedEncoding = durationFeatures.transposed(0, 2, 1).matmul(alignmentTarget)
        let (f0Prediction, nPrediction) = prosodyPredictor.predict(x: alignedEncoding, s: globalStyle)

        // Step 8: Text encoding + alignment
        let textEncoding = textEncoder(paddedInputIds, inputLengths: inputLengths, m: textMask)
        let asrFeatures = MLX.matmul(textEncoding, alignmentTarget)

        // Step 9: Decode to audio
        let audio = decoder(asr: asrFeatures, f0Curve: f0Prediction, n: nPrediction, s: acousticStyle)[0]
        return audio
    }

    // MARK: - Voice Loading

    /// Load a voice embedding from the model directory.
    public func loadVoice(named name: String) throws -> MLXArray {
        if let cached = voiceCache[name] { return cached }

        guard let dir = modelDirectory else {
            throw KokoroError.voiceNotFound(name)
        }

        let voicePath = dir.appendingPathComponent("voices/\(name).safetensors")
        guard FileManager.default.fileExists(atPath: voicePath.path) else {
            throw KokoroError.voiceNotFound(name)
        }

        let voiceWeights = try MLX.loadArrays(url: voicePath)
        guard let voiceArray = voiceWeights["voice"] ?? voiceWeights.values.first else {
            throw KokoroError.voiceNotFound(name)
        }

        voiceCache[name] = voiceArray
        return voiceArray
    }

    /// List available voice names from the model directory.
    public func availableVoices() -> [String] {
        guard let dir = modelDirectory else { return [] }
        let voicesDir = dir.appendingPathComponent("voices")
        guard let files = try? FileManager.default.contentsOfDirectory(atPath: voicesDir.path) else {
            return []
        }
        return files
            .filter { $0.hasSuffix(".safetensors") }
            .map { String($0.dropLast(".safetensors".count)) }
            .sorted()
    }

    // MARK: - Text Processor

    /// Set or replace the text processor used to convert plain text to phonemes.
    ///
    /// When a text processor is set, `generate(text:)` will run the processor
    /// before synthesis. When nil, input text is expected to be pre-phonemized IPA.
    public func setTextProcessor(_ processor: TextProcessor?) {
        self.textProcessor = processor
    }

    // MARK: - Alignment

    private func createAlignmentTarget(durations: MLXArray, batchSize: Int) -> MLXArray {
        let indices = MLX.concatenated(
            durations.enumerated().map { index, duration in
                let frameCount: Int = duration.item()
                return MLX.repeated(MLXArray([index]), count: frameCount)
            }
        )

        let totalFrames = indices.shape[0]
        var alignmentArray = [Float](repeating: 0.0, count: totalFrames * batchSize)
        for frame in 0..<totalFrames {
            let phonemeIndex: Int = indices[frame].item()
            alignmentArray[phonemeIndex * totalFrames + frame] = 1.0
        }

        return MLXArray(alignmentArray)
            .reshaped([batchSize, totalFrames])
            .expandedDimensions(axis: 0)
    }

    // MARK: - Weight Sanitization

    /// Sanitize weight keys from HuggingFace format for Kokoro model construction.
    static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key.hasPrefix("bert") {
                if key.contains("position_ids") { continue }
                sanitized[key] = value

            } else if key.hasPrefix("predictor") {
                if key.contains("F0_proj.weight") || key.contains("N_proj.weight") {
                    sanitized[key] = value.transposed(0, 2, 1)
                } else if key.contains("weight_v") {
                    sanitized[key] = needsTranspose(value) ? value.transposed(0, 2, 1) : value
                } else {
                    sanitized[key] = value
                }

            } else if key.hasPrefix("text_encoder") {
                if key.contains("weight_v") {
                    sanitized[key] = needsTranspose(value) ? value.transposed(0, 2, 1) : value
                } else {
                    sanitized[key] = value
                }

            } else if key.hasPrefix("decoder") {
                if key.contains("noise_convs"), key.hasSuffix(".weight") {
                    sanitized[key] = value.transposed(0, 2, 1)
                } else if key.contains("weight_v") {
                    sanitized[key] = needsTranspose(value) ? value.transposed(0, 2, 1) : value
                } else {
                    sanitized[key] = value
                }
            }
        }
        return sanitized
    }

    /// Check if a 3D weight needs transposition (not in canonical [outCh, kH, kW] form).
    private static func needsTranspose(_ arr: MLXArray) -> Bool {
        guard arr.shape.count == 3 else { return false }
        let (o, h, w) = (arr.shape[0], arr.shape[1], arr.shape[2])
        let isCorrect = (o >= h) && (o >= w) && (h == w)
        return !isCorrect
    }

    // MARK: - Factory

    /// Load a pretrained Kokoro model from HuggingFace Hub.
    ///
    /// - Parameters:
    ///   - modelRepo: HuggingFace repository ID (e.g., "mlx-community/Kokoro-82M-bf16").
    ///   - textProcessor: Optional text processor for converting plain text to phonemes.
    ///     When nil, input text is expected to be pre-phonemized IPA.
    ///   - cache: HuggingFace Hub cache configuration.
    public static func fromPretrained(
        _ modelRepo: String,
        textProcessor: TextProcessor? = nil,
        cache: HubCache = .default
    ) async throws -> KokoroModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(modelRepo)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            hfToken: hfToken,
            cache: cache
        )

        // Load config
        let configURL = modelDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw KokoroError.configNotFound
        }
        let config = try KokoroConfig.load(from: configURL)

        // Find and load weights
        let weightsURL: URL
        let safetensorsFiles = try FileManager.default.contentsOfDirectory(atPath: modelDir.path)
            .filter { $0.hasSuffix(".safetensors") && !$0.contains("voices") }

        if let mainWeights = safetensorsFiles.first(where: { $0.contains("kokoro") }) {
            weightsURL = modelDir.appendingPathComponent(mainWeights)
        } else if let firstWeights = safetensorsFiles.first {
            weightsURL = modelDir.appendingPathComponent(firstWeights)
        } else {
            throw KokoroError.weightsNotFound
        }

        let rawWeights = try MLX.loadArrays(url: weightsURL)
        let weights = sanitizeWeights(rawWeights)

        return KokoroModel(
            config: config, weights: weights,
            modelDirectory: modelDir, textProcessor: textProcessor
        )
    }
}
