import AVFoundation
import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXNN

/// Offline 24 kHz mono audio tokenizer with twenty residual codebooks.
///
/// Waveform arrays use `(samples,)` or `(samples, channels)`. Audio codes use
/// `(codebooks, frames)`; decoding any nonempty codebook prefix is supported.
public final class MiMoAudioTokenizer: Module {
    public static let defaultRepository = "XiaomiMiMo/MiMo-Audio-Tokenizer"
    public let config: MiMoAudioTokenizerConfig
    @ModuleInfo var encoder: MiMoTokenizerEncoder
    @ModuleInfo var decoder: MiMoTokenizerDecoder

    public var sampleRate: Int { config.samplingRate }
    public var numQuantizers: Int { config.numQuantizers }

    public init(config: MiMoAudioTokenizerConfig = MiMoAudioTokenizerConfig()) throws {
        try config.validate()
        self.config = config
        _encoder.wrappedValue = MiMoTokenizerEncoder(config)
        _decoder.wrappedValue = MiMoTokenizerDecoder(config)
    }

    /// Encode a waveform, averaging time-major channels to mono and resampling when needed.
    /// An explicit sample rate is required for in-memory audio.
    public func encode(_ waveform: MLXArray, sampleRate: Int? = nil,
                       numQuantizers: Int? = nil) throws -> MLXArray {
        guard let sampleRate, sampleRate > 0 else {
            throw MiMoAudioTokenizerError.invalidInput("A positive sampleRate is required for waveform arrays")
        }
        guard waveform.ndim == 1 || (waveform.ndim == 2 && waveform.dim(1) > 0), waveform.size > 0 else {
            throw MiMoAudioTokenizerError.invalidInput("Expected nonempty (samples,) or (samples, channels) audio")
        }
        var audio = waveform.asType(.float32)
        if audio.ndim == 2 { audio = mean(audio, axis: -1) }
        guard isFinite(audio).all().item(Bool.self) else {
            throw MiMoAudioTokenizerError.invalidInput("Audio contains non-finite samples")
        }
        audio = try resampleAudio(audio, from: sampleRate, to: self.sampleRate)
        guard audio.size > 0 else { throw MiMoAudioTokenizerError.invalidInput("Resampled audio is empty") }
        return try encodeMels(mimoLogMelSpectrogram(audio, config: config), numQuantizers: numQuantizers)
    }

    public func encode(_ audioURL: URL, numQuantizers: Int? = nil) throws -> MLXArray {
        let file = try AVAudioFile(forReading: audioURL, commonFormat: .pcmFormatFloat32, interleaved: false)
        guard file.length > 0, file.length <= Int64(UInt32.max),
              let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                             frameCapacity: AVAudioFrameCount(file.length)) else {
            throw MiMoAudioTokenizerError.invalidInput("Expected a nonempty audio file that fits in memory")
        }
        try file.read(into: buffer)
        guard let channels = buffer.floatChannelData, buffer.format.channelCount > 0 else {
            throw MiMoAudioTokenizerError.invalidInput("Cannot read floating-point audio samples")
        }
        let frames = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)
        var mono = Array(repeating: Float(0), count: frames)
        for channel in 0..<channelCount {
            for frame in 0..<frames { mono[frame] += channels[channel][frame] }
        }
        if channelCount > 1 { mono = mono.map { $0 / Float(channelCount) } }
        return try encode(MLXArray(mono), sampleRate: Int(buffer.format.sampleRate), numQuantizers: numQuantizers)
    }

    /// Encode exact model log-magnitude mel features `(frames, nMels)`.
    public func encodeMels(_ mels: MLXArray, numQuantizers: Int? = nil,
                           segmentSize: Int = 6000) throws -> MLXArray {
        let count = numQuantizers ?? self.numQuantizers
        guard (1...self.numQuantizers).contains(count) else {
            throw MiMoAudioTokenizerError.invalidInput("numQuantizers is outside the codec's codebook range")
        }
        guard mels.ndim == 2, mels.dim(0) > 0, mels.dim(1) == config.nMels,
              segmentSize > 0, isFinite(mels).all().item(Bool.self) else {
            throw MiMoAudioTokenizerError.invalidInput("Expected nonempty finite (frames, nMels) features and positive segmentSize")
        }
        let paddedLength = min(mels.dim(0), segmentSize)
        var chunks: [MLXArray] = []
        for start in stride(from: 0, to: mels.dim(0), by: segmentSize) {
            let length = min(segmentSize, mels.dim(0) - start)
            let valid = (length + 3 - config.kernelSize + 2 - config.kernelSize) / config.strideSize + 1
            guard valid > 0 else { throw MiMoAudioTokenizerError.invalidInput("Too few mel frames for the configured encoder") }
            var part = mels[start..<(start + length)]
            if length < paddedLength { part = padded(part, widths: [.init((0, paddedLength - length)), .init(0)]) }
            let features = encoder(part.expandedDimensions(axis: 0), length: length)[0]
            let codes = encoder.quantizer.encode(features, count: count)
            eval(codes)
            chunks.append(codes)
        }
        return concatenated(chunks, axis: -1)
    }

    /// Decode integer `(codebooks, frames)` codes to a mono waveform at `sampleRate`.
    /// `length`, when given, trims padding using a sample count at the codec rate.
    public func decode(_ codes: MLXArray, segmentSize: Int = 1500,
                       length: Int? = nil) throws -> MLXArray {
        guard codes.ndim == 2, (1...numQuantizers).contains(codes.dim(0)), codes.dtype.isInteger else {
            throw MiMoAudioTokenizerError.invalidInput("Expected integer codes with shape (codebooks, frames)")
        }
        guard segmentSize > 0 && (length == nil || length! >= 0) else {
            throw MiMoAudioTokenizerError.invalidInput("Invalid segment size or output length")
        }
        if codes.dim(1) == 0 { return MLXArray.zeros([0], dtype: .float32) }
        for index in 0..<codes.dim(0) {
            let values = codes[index]
            guard !(values .< 0).any().item(Bool.self),
                  !(values .>= config.codebookSize[index]).any().item(Bool.self) else {
                throw MiMoAudioTokenizerError.invalidInput("Audio code is outside its codebook (padding is not audio)")
            }
        }
        var parts: [MLXArray] = []
        for start in stride(from: 0, to: codes.dim(1), by: segmentSize) {
            let end = min(codes.dim(1), start + segmentSize)
            let features = encoder.quantizer.decode(codes[0..., start..<end])
            let audio = decoder(features.expandedDimensions(axis: 0))[0].asType(.float32)
            eval(audio)
            parts.append(audio)
        }
        let result = concatenated(parts)
        return length.map { result[0..<min($0, result.size)] } ?? result
    }

    /// Adapt original Xiaomi tensor names/layouts; prepared MLX tensors pass through.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        if config.weightsFormat == "mlx" { return weights }
        var result: [String: MLXArray] = [:]
        for (sourceKey, sourceValue) in weights {
            var key = sourceKey, value = sourceValue
            let fields = key.split(separator: ".").map(String.init)
            if fields.count == 7 && fields.prefix(4) == ["encoder", "quantizer", "vq", "layers"]
                && fields[5] == "_codebook" {
                if ["cluster_size", "embed_avg", "inited"].contains(fields[6]) { continue }
                if fields[6] == "embed" { key = "encoder.quantizer.codebooks.\(fields[4]).weight" }
            } else if key.hasSuffix("weight") && value.ndim == 3 {
                value = key.contains(".dconv") ? value.transposed(1, 2, 0) : value.transposed(0, 2, 1)
            }
            key = key.replacingOccurrences(of: "encoder.down_sample_layer.0.",
                                           with: "encoder.down_sample_layer.layers.0.")
            result[key] = value
        }
        return result
    }

    public static func fromPretrained(_ source: String = defaultRepository, dtype: DType? = .bfloat16,
                                      hfToken: String? = nil, cache: HubCache = .default) async throws -> MiMoAudioTokenizer {
        let path = NSString(string: source).expandingTildeInPath
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) {
            guard isDirectory.boolValue else {
                throw MiMoAudioTokenizerError.missingDirectory(URL(fileURLWithPath: path))
            }
            return try fromModelDirectory(URL(fileURLWithPath: path), dtype: dtype)
        }
        if path.hasPrefix("/") || source.hasPrefix(".") || source.hasPrefix("~") {
            throw MiMoAudioTokenizerError.missingDirectory(URL(fileURLWithPath: path))
        }
        guard let repoID = Repo.ID(rawValue: source) else {
            return try fromModelDirectory(URL(fileURLWithPath: path), dtype: dtype)
        }
        let directory = try await ModelUtils.resolveOrDownloadModel(repoID: repoID, requiredExtension: ".safetensors",
                                                                    hfToken: hfToken, cache: cache)
        return try fromModelDirectory(directory, dtype: dtype)
    }

    public static func fromModelDirectory(_ directory: URL, dtype: DType? = .bfloat16) throws -> MiMoAudioTokenizer {
        if let dtype, ![DType.float32, .float16, .bfloat16].contains(dtype) {
            throw MiMoAudioTokenizerError.invalidConfiguration("inference dtype must be floating point")
        }
        let config = try JSONDecoder().decode(MiMoAudioTokenizerConfig.self,
                                              from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw MiMoAudioTokenizerError.missingWeights(directory) }
        var source: [String: MLXArray] = [:]
        for file in files {
            for (name, tensor) in try loadArrays(url: file) {
                guard source[name] == nil else {
                    throw MiMoAudioTokenizerError.invalidConfiguration("duplicate checkpoint tensor \(name)")
                }
                source[name] = tensor
            }
        }
        let model = try MiMoAudioTokenizer(config: config)
        var weights = model.sanitize(weights: source)
        for (key, value) in weights {
            // Match the reference's cast-then-restore sequence, including BF16
            // rounding of RVQ tables before their FP32 lookup/accumulation.
            let cast = dtype.map { value.asType($0) } ?? value
            weights[key] = key.contains(".quantizer.") ? cast.asType(.float32) : cast
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.train(false)
        eval(model.parameters())
        return model
    }
}
