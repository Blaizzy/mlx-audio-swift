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
    private let fixedFrames: Int
    private let subsamplingFactor: Int
    private let inputName: String
    private let outputName: String

    public init(
        modelURL: URL,
        featIn: Int,
        fixedFrames: Int,
        subsamplingFactor: Int,
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
    }

    /// Matches `ParakeetModel.computeEncodedLengths`: `floor((L-1)/2)+1`, log2(factor) times.
    static func subsampledLength(frames: Int, subsamplingFactor: Int) -> Int {
        var l = frames
        let steps = Int(log2(Double(subsamplingFactor)))
        for _ in 0..<steps { l = (l - 1) / 2 + 1 }
        return l
    }

    private func encodedLength(for frames: Int) -> Int {
        Self.subsampledLength(frames: frames, subsamplingFactor: subsamplingFactor)
    }

    /// Encode one chunk. `features`: `[1, T, featIn]` (any float dtype).
    /// Returns `(encoded [1, T', dModel], lengths [1])`, dtype = `outputDType`.
    public func encode(_ features: MLXArray, outputDType: DType) throws -> (MLXArray, MLXArray) {
        let trueFrames = features.shape[1]
        let clamped = min(trueFrames, fixedFrames)

        var mel = features.asType(.float32)
        if trueFrames < fixedFrames {
            mel = padded(mel, widths: [.init((0, 0)), .init((0, fixedFrames - trueFrames)), .init((0, 0))])
        } else if trueFrames > fixedFrames {
            mel = mel[0..., 0..<fixedFrames, 0...]
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
