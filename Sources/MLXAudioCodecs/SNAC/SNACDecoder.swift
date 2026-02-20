import Foundation
import MLX
import MLXNN
import HuggingFace
import MLXAudioCore



// MARK: - SNAC Model

public class SNAC: Module {
    public let samplingRate: Int
    public let encoderDim: Int
    public let encoderRates: [Int]
    public let decoderDim: Int
    public let decoderRates: [Int]
    public let latentDim: Int
    public let hopLength: Int
    public let nCodebooks: Int
    public let codebookSize: Int
    public let codebookDim: Int
    public let vqStrides: [Int]
    public let attnWindowSize: Int?

    let encoder: Encoder
    let quantizer: ResidualVectorQuantize
    let decoder: Decoder

    public init(
        samplingRate: Int = 44100,
        encoderDim: Int = 64,
        encoderRates: [Int] = [3, 3, 7, 7],
        latentDim: Int? = nil,
        decoderDim: Int = 1536,
        decoderRates: [Int] = [7, 7, 3, 3],
        attnWindowSize: Int? = 32,
        codebookSize: Int = 4096,
        codebookDim: Int = 8,
        vqStrides: [Int] = [8, 4, 2, 1],
        noise: Bool = true,
        depthwise: Bool = true
    ) {
        self.samplingRate = samplingRate
        self.encoderDim = encoderDim
        self.encoderRates = encoderRates
        self.decoderDim = decoderDim
        self.decoderRates = decoderRates

        // Calculate latent_dim if not provided
        let calculatedLatentDim = latentDim ?? (encoderDim * Int(pow(2.0, Double(encoderRates.count))))
        self.latentDim = calculatedLatentDim

        // Calculate hop_length (product of encoder rates)
        self.hopLength = encoderRates.reduce(1, *)

        self.nCodebooks = vqStrides.count
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        self.vqStrides = vqStrides
        self.attnWindowSize = attnWindowSize

        self.encoder = Encoder(
            dModel: encoderDim,
            strides: encoderRates,
            depthwise: depthwise,
            attnWindowSize: attnWindowSize
        )

        self.quantizer = ResidualVectorQuantize(
            inputDim: calculatedLatentDim,
            codebookSize: codebookSize,
            codebookDim: codebookDim,
            vqStrides: vqStrides
        )

        self.decoder = Decoder(
            inputChannel: calculatedLatentDim,
            channels: decoderDim,
            rates: decoderRates,
            noise: noise,
            depthwise: depthwise,
            attnWindowSize: attnWindowSize
        )
    }

    public func preprocess(_ audioData: MLXArray) -> MLXArray {
        let length = audioData.shape[audioData.ndim - 1]

        // Calculate LCM of all vq_strides
        var lcmValue = vqStrides[0]
        for i in 1..<vqStrides.count {
            lcmValue = lcm(lcmValue, vqStrides[i])
        }

        // Include attention window size in LCM calculation if present
        if let attnWindowSize = attnWindowSize {
            lcmValue = lcm(lcmValue, attnWindowSize)
        }

        let padTo = hopLength * lcmValue
        let rightPad = Int(ceil(Double(length) / Double(padTo))) * padTo - length

        // Pad the audio data: [(0, 0), (0, 0), (0, right_pad)]
        return audioData.padded([(0, 0), (0, 0), (0, rightPad)])
    }

    public func callAsFunction(_ audioData: MLXArray) -> (MLXArray, [MLXArray]) {
        let length = audioData.shape[audioData.ndim - 1]
        let preprocessed = preprocess(audioData)

        let z = encoder(preprocessed)
        let (zQ, codes) = quantizer(z)
        let audioHat = decoder(zQ)

        // Trim to original length
        let trimmed = audioHat[.ellipsis, 0..<length]
        return (trimmed, codes)
    }

    public func encode(_ audioData: MLXArray) -> [MLXArray] {
        let preprocessed = preprocess(audioData)
        let z = encoder(preprocessed)
        let (_, codes) = quantizer(z)
        return codes
    }

    public func decode(_ codes: [MLXArray]) -> MLXArray {
        let zQ = quantizer.fromCodes(codes)
        let audioHat = decoder(zQ)
        return audioHat
    }

    // MARK: - Loading Methods

    public static func fromConfig(_ configPath: URL) throws -> SNAC {
        let data = try Data(contentsOf: configPath)
        let decoder = JSONDecoder()
        let config = try decoder.decode(SNACConfig.self, from: data)

        return SNAC(
            samplingRate: config.samplingRate,
            encoderDim: config.encoderDim,
            encoderRates: config.encoderRates,
            latentDim: config.latentDim,
            decoderDim: config.decoderDim,
            decoderRates: config.decoderRates,
            attnWindowSize: config.attnWindowSize,
            codebookSize: config.codebookSize,
            codebookDim: config.codebookDim,
            vqStrides: config.vqStrides,
            noise: config.noise,
            depthwise: config.depthwise
        )
    }

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> SNAC {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw NSError(domain: "SNAC", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelRepo)"])
        }

        // Check if model is already fully cached (has weight files)
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            cache: cache
        )


        let configPath = modelDir.appendingPathComponent("config.json")
        let weightsPath = modelDir.appendingPathComponent("model.safetensors")

        guard FileManager.default.fileExists(atPath: weightsPath.path) else {
            throw SNACError.modelNotFound("Could not find model at \(weightsPath.path)")
        }

        let snac = try fromConfig(configPath)

        let weights = try loadArrays(url: weightsPath)
        try snac.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(snac)

        return snac
    }
}

extension SNAC: AudioCodecModel {
    public typealias EncodedAudio = [MLXArray]

    public var codecSampleRate: Double? { Double(samplingRate) }

    public func encodeAudio(_ waveform: MLXArray) -> [MLXArray] {
        encode(waveform)
    }

    public func decodeAudio(_ input: [MLXArray]) -> MLXArray {
        decode(input)
    }
}

// MARK: - Helper Functions

func lcm(_ a: Int, _ b: Int) -> Int {
    return abs(a * b) / gcd(a, b)
}

func gcd(_ a: Int, _ b: Int) -> Int {
    var a = a
    var b = b
    while b != 0 {
        let temp = b
        b = a % b
        a = temp
    }
    return a
}
// MARK: - Error Types

public enum SNACError: Error {
    case modelNotFound(String)
    case configLoadError(String)
    case weightsLoadError(String)
}

// MARK: - Extension for Padded

extension MLXArray {
    func padded(_ padWidths: [(Int, Int)]) -> MLXArray {
        // Convert [(Int, Int)] to [IntOrPair]
        let paddingArray = padWidths.map { IntOrPair($0) }
        return MLX.padded(self, widths: paddingArray)
    }
}
