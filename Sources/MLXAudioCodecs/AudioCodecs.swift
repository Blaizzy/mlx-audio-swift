import Foundation
import HuggingFace
import MLX
import MLXAudioCore

public enum AudioCodecError: Error, LocalizedError {
    case unsupportedModel(String)
    case missingDirectory(String)
    case invalidWaveform(String)
    case invalidOutput(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedModel(let model):
            "Unsupported roundtrip audio codec: \(model)"
        case .missingDirectory(let path):
            "Codec directory not found: \(path)"
        case .invalidWaveform(let message), .invalidOutput(let message):
            message
        }
    }
}

/// A roundtrip codec with a common `(batch, samples, channels)` waveform layout.
/// Use the concrete model for code/latent access and model-specific controls.
public struct AnyAudioCodec {
    public let sampleRate: Int
    public let audioChannels: Int
    private let reconstruction: (MLXArray) throws -> MLXArray

    init(
        sampleRate: Int,
        audioChannels: Int = 1,
        reconstruction: @escaping (MLXArray) throws -> MLXArray
    ) {
        self.sampleRate = sampleRate
        self.audioChannels = audioChannels
        self.reconstruction = reconstruction
    }

    /// Reconstruct audio already resampled to `sampleRate`, trimming framing padding.
    public func reconstruct(_ waveform: MLXArray) throws -> MLXArray {
        guard waveform.ndim == 3, waveform.dim(0) > 0, waveform.dim(1) > 0,
              waveform.dim(2) == audioChannels else {
            throw AudioCodecError.invalidWaveform(
                "Expected nonempty waveform (batch, samples, \(audioChannels)), got \(waveform.shape)"
            )
        }
        let result = try reconstruction(waveform.asType(.float32))
        guard result.ndim == 3, result.dim(0) == waveform.dim(0),
              result.dim(2) == audioChannels, result.dim(1) >= waveform.dim(1) else {
            throw AudioCodecError.invalidOutput(
                "Codec returned \(result.shape) for waveform \(waveform.shape)"
            )
        }
        return result[0..., 0..<waveform.dim(1), 0...].asType(.float32)
    }
}

/// Shared loading for codecs that support waveform reconstruction.
public enum AudioCodecs {
    enum Family: String {
        case dacvae, encodec, snac, mimi, descript, fishS1DAC = "fish_s1_dac"
        case mimoAudioTokenizer = "mimo_audio_tokenizer"
        case nemotronVoiceChat = "nemotron_voicechat_codec"
    }

    /// Load a local directory or Hugging Face repository using its codec metadata.
    public static func loadModel(
        modelRepo: String,
        cache: HubCache = .default
    ) async throws -> AnyAudioCodec {
        let expanded = NSString(string: modelRepo).expandingTildeInPath
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: expanded, isDirectory: &isDirectory) {
            guard isDirectory.boolValue else { throw AudioCodecError.missingDirectory(expanded) }
            return try fromModelDirectory(URL(fileURLWithPath: expanded))
        }
        if expanded.hasPrefix("/") || modelRepo.hasPrefix(".") || modelRepo.hasPrefix("~") {
            throw AudioCodecError.missingDirectory(expanded)
        }
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioCodecError.unsupportedModel(modelRepo)
        }

        // Moshi repositories contain a language-model config alongside Mimi's
        // named tokenizer checkpoint. Preserve that explicit artifact contract.
        if inferredFamily(modelRepo) == .mimi {
            let model = try await Mimi.fromPretrained(repoId: modelRepo, cache: cache, progressHandler: { _ in })
            return wrap(model)
        }

        // Fetch only the config to select the model. In particular, selecting a
        // codec embedded in a large speech model must not download that whole model.
        let configURL = try await HubClient(cache: cache).downloadFile(at: "config.json", from: repoID)
        let config = try readConfig(configURL)
        let family = try resolveFamily(config: config, source: modelRepo)
        if family == .mimoAudioTokenizer {
            return wrap(try await MiMoAudioTokenizer.fromPretrained(modelRepo, cache: cache))
        }
        if family == .nemotronVoiceChat {
            return wrap(try await NemotronVoiceChatCodec.fromPretrained(modelRepo, cache: cache))
        }
        let directory = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            cache: cache
        )
        return try load(family, directory: directory)
    }

    /// Load without Hub access, including arbitrarily named or symlinked directories.
    public static func fromModelDirectory(_ directory: URL) throws -> AnyAudioCodec {
        let configURL = directory.appendingPathComponent("config.json")
        let config = FileManager.default.fileExists(atPath: configURL.path)
            ? try readConfig(configURL) : [:]
        let source = directory.lastPathComponent
        let family: Family
        if config.isEmpty,
           FileManager.default.fileExists(
               atPath: directory.appendingPathComponent("tokenizer-e351c8d8-checkpoint125.safetensors").path
           ) {
            family = .mimi
        } else {
            family = try resolveFamily(config: config, source: source)
        }
        return try load(family, directory: directory)
    }

    static func resolveFamily(config: [String: Any], source: String) throws -> Family {
        if let declared = config["model_type"] as? String,
           !declared.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let normalized = declared.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if let family = Family(rawValue: normalized) { return family }
            if normalized == "nemotron_voicechat" { return .nemotronVoiceChat }
            if normalized == "dac" { return .descript }
            if normalized == "fish-s1-dac" { return .fishS1DAC }
            throw AudioCodecError.unsupportedModel(declared)
        }
        // Older codec configs predate model_type. Prefer distinguishing config
        // fields to repository names, which may be arbitrary for local models.
        if config["encoder_attention_heads"] != nil && config["vocoder_attention_heads"] != nil {
            return .mimoAudioTokenizer
        }
        if config["codec_config"] is [String: Any]
            || ((config["model"] as? [String: Any])?["speech_generation"] is [String: Any])
            || ((config["base_channels"] != nil || config["base_hidden_size"] != nil)
                && (config["downsample_rates"] != nil || config["rates"] != nil)) {
            return .nemotronVoiceChat
        }
        if config["vq_strides"] != nil { return .snac }
        if config["target_bandwidths"] != nil { return .encodec }
        if config["encoder_rates"] != nil {
            if config["downsample_factor"] is [Any] || config["semantic_codebook_size"] != nil { return .fishS1DAC }
            if config["mean"] != nil || config["std"] != nil { return .dacvae }
            if config["n_codebooks"] != nil { return .descript }
        }
        if let inferred = inferredFamily(source) { return inferred }
        throw AudioCodecError.unsupportedModel(source)
    }

    static func inferredFamily(_ source: String) -> Family? {
        let name = source.lowercased()
        if name.contains("mimo-audio-tokenizer") || name.contains("mimo_audio_tokenizer") { return .mimoAudioTokenizer }
        if name.contains("nemotron") && name.contains("voicechat") { return .nemotronVoiceChat }
        if name.contains("dacvae") { return .dacvae }
        if name.contains("encodec") { return .encodec }
        if name.contains("snac") { return .snac }
        if name.contains("mimi") || name.contains("moshiko") { return .mimi }
        if name.contains("descript") { return .descript }
        if name.contains("fish_s1_dac") || name.contains("fish-s1-dac") { return .fishS1DAC }
        return nil
    }

    private static func readConfig(_ url: URL) throws -> [String: Any] {
        guard let config = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw AudioCodecError.unsupportedModel(url.path)
        }
        return config
    }

    private static func load(_ family: Family, directory: URL) throws -> AnyAudioCodec {
        switch family {
        case .dacvae:
            let model = try DACVAE.fromModelDirectory(directory)
            return AnyAudioCodec(sampleRate: model.sampleRate) { model.reconstruct($0) }
        case .encodec:
            let model = try Encodec.fromModelDirectory(directory)
            return AnyAudioCodec(sampleRate: model.samplingRate, audioChannels: model.config.audioChannels) {
                model.reconstruct($0)
            }
        case .snac:
            let model = try SNAC.fromModelDirectory(directory)
            return AnyAudioCodec(sampleRate: model.samplingRate) {
                model.reconstruct($0.transposed(0, 2, 1)).transposed(0, 2, 1)
            }
        case .mimi:
            return wrap(try Mimi.fromModelDirectory(directory))
        case .descript:
            let model = try DescriptDAC.fromModelDirectory(directory)
            return AnyAudioCodec(sampleRate: model.sampleRate) { model.reconstruct($0) }
        case .fishS1DAC:
            let model = try FishS1DAC.fromModelDirectory(directory)
            return AnyAudioCodec(sampleRate: model.sampleRate) { model.reconstruct($0) }
        case .mimoAudioTokenizer:
            return wrap(try MiMoAudioTokenizer.fromModelDirectory(directory))
        case .nemotronVoiceChat:
            return wrap(try NemotronVoiceChatCodec.fromModelDirectory(directory))
        }
    }

    private static func wrap(_ model: Mimi) -> AnyAudioCodec {
        AnyAudioCodec(sampleRate: Int(model.sampleRate.rounded()), audioChannels: model.cfg.channels) {
            model.reconstruct($0.transposed(0, 2, 1)).transposed(0, 2, 1)
        }
    }

    private static func wrap(_ model: MiMoAudioTokenizer) -> AnyAudioCodec {
        AnyAudioCodec(sampleRate: model.sampleRate) { waveform in
            var outputs: [MLXArray] = []
            for batch in 0..<waveform.dim(0) {
                let codes = try model.encode(waveform[batch], sampleRate: model.sampleRate)
                outputs.append(try model.decode(codes, length: waveform.dim(1)))
            }
            return stacked(outputs, axis: 0).expandedDimensions(axis: -1)
        }
    }

    private static func wrap(_ model: NemotronVoiceChatCodec) -> AnyAudioCodec {
        AnyAudioCodec(sampleRate: model.sampleRate) { model.reconstruct($0) }
    }
}
