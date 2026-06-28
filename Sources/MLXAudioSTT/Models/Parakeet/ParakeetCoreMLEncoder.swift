#if canImport(CoreML)
import CoreML
import Foundation
import MLX

/// Drop-in CoreML/ANE replacement for the MLX Conformer encoder. The model is
/// fixed-shape because ANE requires it (RangeDim → 0% residency), so each chunk's mel
/// is padded to `fixedFrames` and the output cropped back to the true subsampled length.
public final class ParakeetCoreMLEncoder: @unchecked Sendable {
    private let model: MLModel
    private let featIn: Int
    /// The fixed mel-frame count the model was converted at (e.g. 1000 = 10 s). Callers use
    /// this to keep their chunking ≤ this length, since longer mel is cropped.
    public let fixedFrames: Int
    private let subsamplingFactor: Int
    private let inputName: String
    private let outputName: String
    /// Expected encoder output width (`d_model`); when set, `encode` rejects a model whose
    /// output doesn't match the loaded checkpoint instead of returning mis-shaped features.
    private let expectedDModel: Int?
    /// One subsampling stage on a length. Parakeet: `(L-1)/2+1`; encoders with different conv
    /// padding (e.g. Nemotron's pad-before-stride `floor(L/2)+1`) inject their own so the ANE
    /// output is cropped to the right valid width.
    private let subsampledLengthStep: (Int) -> Int

    public init(
        modelURL: URL,
        featIn: Int,
        fixedFrames: Int,
        subsamplingFactor: Int,
        dModel: Int? = nil,
        subsampledLengthStep: (@Sendable (Int) -> Int)? = nil,
        computeUnits: MLComputeUnits = .all,
        inputName: String = "features",
        outputName: String = "encoded"
    ) throws {
        let compiledURL: URL
        if modelURL.pathExtension == "mlmodelc" {
            compiledURL = modelURL
        } else {
            compiledURL = try MLModel.compileModel(at: modelURL)
        }
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)
        self.featIn = featIn
        self.fixedFrames = fixedFrames
        self.subsamplingFactor = subsamplingFactor
        self.inputName = inputName
        self.outputName = outputName
        self.expectedDModel = dModel
        self.subsampledLengthStep = subsampledLengthStep ?? { ($0 - 1) / 2 + 1 }
    }

    /// Parakeet default: `floor((L-1)/2)+1`, log2(factor) times. `step` overrides the per-stage
    /// rule for encoders whose conv padding differs (e.g. Nemotron's `floor(L/2)+1`).
    static func subsampledLength(
        frames: Int,
        subsamplingFactor: Int,
        step: (Int) -> Int = { ($0 - 1) / 2 + 1 }
    ) -> Int {
        var l = frames
        let steps = Int(log2(Double(subsamplingFactor)))
        for _ in 0..<steps { l = step(l) }
        return l
    }

    private func encodedLength(for frames: Int) -> Int {
        Self.subsampledLength(frames: frames, subsamplingFactor: subsamplingFactor, step: subsampledLengthStep)
    }

    /// Encode one chunk. `features`: `[1, T, featIn]` (any float dtype).
    /// Returns `(encoded [1, T', dModel], lengths [1])`, dtype = `outputDType`.
    public func encode(_ features: MLXArray, outputDType: DType) throws -> (MLXArray, MLXArray) {
        // Fixed-shape ANE model: it only handles a single chunk that fits the window.
        // Throw (rather than silently truncate a long chunk or scramble a multi-item batch)
        // so callers — which invoke `encode` via `try?` — fall back to the MLX encoder and
        // still produce a complete, correct result.
        guard features.shape[0] == 1 else {
            throw STTError.invalidInput("CoreML encoder handles batch size 1 (got \(features.shape[0])); falling back to MLX")
        }
        let trueFrames = features.shape[1]
        guard trueFrames <= fixedFrames else {
            throw STTError.invalidInput(
                "chunk of \(trueFrames) mel frames exceeds the fixed CoreML window of \(fixedFrames); "
                    + "use a shorter --chunk-duration or the MLX encoder")
        }
        let clamped = trueFrames

        var mel = features.asType(.float32)
        if trueFrames < fixedFrames {
            mel = padded(mel, widths: [.init((0, 0)), .init((0, fixedFrames - trueFrames)), .init((0, 0))])
        }
        // mel is [1, fixedFrames, featIn] row-major; CoreML wants [1, featIn, fixedFrames].
        let melFlat = mel.asArray(Float.self)  // index = t * featIn + f
        let input = try MLMultiArray(shape: [1, NSNumber(value: featIn), NSNumber(value: fixedFrames)], dataType: .float32)
        input.dataPointer.withMemoryRebound(to: Float.self, capacity: featIn * fixedFrames) { dst in
            for t in 0..<fixedFrames {
                for f in 0..<featIn {
                    dst[f * fixedFrames + t] = melFlat[t * featIn + f]
                }
            }
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: input)])
        let out = try model.prediction(from: provider)
        guard let enc = out.featureValue(for: outputName)?.multiArrayValue else {
            throw STTError.invalidInput("CoreML encoder produced no '\(outputName)' output")
        }

        let dModel = enc.shape[1].intValue
        let tFull = enc.shape[2].intValue
        guard expectedDModel == nil || dModel == expectedDModel else {
            throw STTError.invalidInput(
                "CoreML encoder output dim \(dModel) ≠ model d_model \(expectedDModel!); this ANE "
                    + "encoder doesn't match the loaded checkpoint — falling back to MLX")
        }
        // ANE outputs are often stride-padded, so honor strides rather than reading the
        // raw buffer sequentially (which would scramble frames).
        let s1 = enc.strides[1].intValue
        let s2 = enc.strides[2].intValue
        let count = dModel * tFull
        let capacity = (dModel - 1) * s1 + (tFull - 1) * s2 + 1
        var encFloats = [Float](repeating: 0, count: count)  // packed [d * tFull + t]
        if enc.dataType == .float16 {
            let p = enc.dataPointer.bindMemory(to: UInt16.self, capacity: capacity)
            for d in 0..<dModel { for t in 0..<tFull { encFloats[d * tFull + t] = Float(Float16(bitPattern: p[d * s1 + t * s2])) } }
        } else {
            let p = enc.dataPointer.bindMemory(to: Float.self, capacity: capacity)
            for d in 0..<dModel { for t in 0..<tFull { encFloats[d * tFull + t] = p[d * s1 + t * s2] } }
        }

        let validLen = encodedLength(for: clamped)
        var encoded = MLXArray(encFloats, [1, dModel, tFull]).transposed(0, 2, 1)
        if validLen < tFull {
            encoded = encoded[0..., 0..<validLen, 0...]
        }
        let lengths = MLXArray([Int32(validLen)]).asType(.int32)
        return (encoded.asType(outputDType), lengths)
    }
}
#endif
