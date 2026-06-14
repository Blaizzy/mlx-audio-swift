#if canImport(CoreML)
import CoreML
import Foundation
import MLX

/// Cache-aware **streaming** CoreML/ANE encoder for Nemotron's FastConformer.
///
/// The model is NeMo's full `cache_aware_stream_step` converted *functionally* — the three
/// streaming caches are explicit inputs **and** outputs (no `MLState`), so we thread them in
/// Swift: feed caches in → read the `new_*` caches out → feed them back next chunk.
///
/// It is fed the **uniform-F** recipe (validated transcript-identical to NeMo native streaming):
/// every chunk is `[preFrames prev-mel ++ newFrames new-mel]` = `fixedFrames`, stride `newFrames`,
/// zeros for the first prepend and the last new-frame tail. A fixed-shape model cannot honor a
/// per-chunk true length, so the feeding — not padding NeMo's variable chunks — is what makes it
/// correct. `cache_aware_stream_step` applies `drop_extra_pre_encoded` internally, so the output is
/// already the valid frames; the caller keeps `ceil(realNew / subsampling)` of them.
public final class NemotronCoreMLStreamingEncoder: @unchecked Sendable {
    private let model: MLModel
    public let featIn: Int
    public let dModel: Int
    /// pre-encode prepend (= NeMo `pre_encode_cache_size[1]`, e.g. 9).
    public let preFrames: Int
    /// new mel frames per chunk (= NeMo `chunk_size[1]`, e.g. 112).
    public let newFrames: Int
    public let subsamplingFactor: Int

    private let layers: Int      // cache_last_channel/time dim 0 (e.g. 24)
    private let attnCache: Int   // cache_last_channel dim 2 (e.g. 70)
    private let convCache: Int   // cache_last_time   dim 3 (e.g. 8)

    private let signalName = "processed_signal"
    private let chInName = "cache_last_channel"
    private let timeInName = "cache_last_time"
    private let lenInName = "cache_last_channel_len"
    private let encodedName = "encoded"
    private let chOutName = "new_cache_last_channel"
    private let timeOutName = "new_cache_last_time"
    private let lenOutName = "new_cache_last_channel_len"

    private var cacheChannel: MLMultiArray
    private var cacheTime: MLMultiArray
    private var cacheLen: MLMultiArray

    /// Mel frames fed per chunk (`preFrames + newFrames`, the model's fixed input length).
    public var fixedFrames: Int { preFrames + newFrames }

    public init(
        modelURL: URL,
        featIn: Int,
        dModel: Int,
        subsamplingFactor: Int,
        preFrames: Int,
        newFrames: Int,
        layers: Int,
        attnCache: Int,
        convCache: Int,
        // Force CPU+ANE, NOT .all: the streaming graph's int32 mask/cache-length ops nudge `.all`
        // to place the WHOLE model on the GPU (anemll-profile: 1.9% ANE) — no power win. With
        // .cpuAndNeuralEngine the conformer is 100% ANE-resident (the few int32 mask ops drop to
        // CPU as one negligible island). The offline encoder has no such masks, so it uses .all.
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) throws {
        let compiledURL = try Self.cachedCompiledModel(at: modelURL)
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)
        self.featIn = featIn
        self.dModel = dModel
        self.subsamplingFactor = subsamplingFactor
        self.preFrames = preFrames
        self.newFrames = newFrames
        self.layers = layers
        self.attnCache = attnCache
        self.convCache = convCache
        self.cacheChannel = try Self.zeroArray([layers, 1, attnCache, dModel], .float16)
        self.cacheTime = try Self.zeroArray([layers, 1, dModel, convCache], .float16)
        self.cacheLen = try MLMultiArray(shape: [1], dataType: .int32)
        self.cacheLen[0] = 0
    }

    /// Compile the `.mlpackage` to `.mlmodelc` once and **cache it** next to the source, so the
    /// ~30 s CoreML/ANE compilation is a one-time **cold start**; later loads reuse the cached
    /// `.mlmodelc` (**hot start**, ~instant). `MLModel.compileModel` alone compiles to a throwaway
    /// temp dir every launch — that is what made every run pay the full compile. Falls back to the
    /// temp compile if the cache location isn't writable.
    private static func cachedCompiledModel(at modelURL: URL) throws -> URL {
        if modelURL.pathExtension == "mlmodelc" { return modelURL }
        let fm = FileManager.default
        let cached = modelURL.deletingPathExtension().appendingPathExtension("mlmodelc")
        func modDate(_ url: URL) -> Date? {
            (try? fm.attributesOfItem(atPath: url.path)[.modificationDate]) as? Date
        }
        if fm.fileExists(atPath: cached.path),
           let c = modDate(cached), let s = modDate(modelURL), c >= s {
            return cached  // hot start: reuse the cached compiled model
        }
        let tmp = try MLModel.compileModel(at: modelURL)  // cold start: compile once
        if fm.fileExists(atPath: cached.path) { try? fm.removeItem(at: cached) }
        do { try fm.moveItem(at: tmp, to: cached); return cached }
        catch { return tmp }  // cache dir not writable -> use the temp compile (no caching)
    }

    /// Reset the three caches to their initial (all-zero) streaming state. Allocates **fresh
    /// contiguous** arrays: after a `step()` the cache vars point at the model's stride-padded
    /// output, which a sequential in-place zero would not fully clear (the padded gaps survive).
    public func reset() {
        cacheChannel = (try? Self.zeroArray([layers, 1, attnCache, dModel], .float16)) ?? cacheChannel
        cacheTime = (try? Self.zeroArray([layers, 1, dModel, convCache], .float16)) ?? cacheTime
        cacheLen = (try? MLMultiArray(shape: [1], dataType: .int32)) ?? cacheLen
        cacheLen[0] = 0
    }

    /// A freshly-allocated, contiguous, all-zero MLMultiArray (contiguous → sequential fill is safe).
    private static func zeroArray(_ shape: [Int], _ dtype: MLMultiArrayDataType) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: dtype)
        let n = a.count
        if dtype == .float16 {
            let p = a.dataPointer.bindMemory(to: UInt16.self, capacity: n)
            for i in 0..<n { p[i] = 0 }
        } else {
            let p = a.dataPointer.bindMemory(to: Float.self, capacity: n)
            for i in 0..<n { p[i] = 0 }
        }
        return a
    }

    /// Encode one uniform-`fixedFrames` mel window `[1, fixedFrames, featIn]` (time-major).
    /// Returns `encoded [1, dModel, T']` (channel-first); the caller crops to valid frames and
    /// transposes. Threads (mutates) the three caches.
    public func step(_ window: MLXArray) throws -> MLXArray {
        let F = fixedFrames
        precondition(window.shape == [1, F, featIn],
                     "stream window must be [1, \(F), \(featIn)], got \(window.shape)")
        // [1, F, featIn] row-major (t*featIn+f) -> processed_signal [1, featIn, F] fp16.
        let melFlat = window.asType(.float32).asArray(Float.self)
        let signal = try MLMultiArray(
            shape: [1, NSNumber(value: featIn), NSNumber(value: F)], dataType: .float16)
        signal.dataPointer.withMemoryRebound(to: UInt16.self, capacity: featIn * F) { dst in
            for t in 0..<F {
                for f in 0..<featIn {
                    dst[f * F + t] = Float16(melFlat[t * featIn + f]).bitPattern
                }
            }
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            signalName: MLFeatureValue(multiArray: signal),
            chInName: MLFeatureValue(multiArray: cacheChannel),
            timeInName: MLFeatureValue(multiArray: cacheTime),
            lenInName: MLFeatureValue(multiArray: cacheLen),
        ])
        let out = try model.prediction(from: provider)

        guard let enc = out.featureValue(for: encodedName)?.multiArrayValue,
              let nch = out.featureValue(for: chOutName)?.multiArrayValue,
              let nt = out.featureValue(for: timeOutName)?.multiArrayValue,
              let nl = out.featureValue(for: lenOutName)?.multiArrayValue
        else {
            throw STTError.invalidInput("CoreML streaming encoder missing an output")
        }
        cacheChannel = nch
        cacheTime = nt
        cacheLen = nl

        // ANE outputs are stride-padded — read by strides, not sequentially.
        let dm = enc.shape[1].intValue
        let tFull = enc.shape[2].intValue
        let s1 = enc.strides[1].intValue
        let s2 = enc.strides[2].intValue
        let capacity = (dm - 1) * s1 + (tFull - 1) * s2 + 1
        var floats = [Float](repeating: 0, count: dm * tFull)
        if enc.dataType == .float16 {
            let p = enc.dataPointer.bindMemory(to: UInt16.self, capacity: capacity)
            for d in 0..<dm { for t in 0..<tFull { floats[d * tFull + t] = Float(Float16(bitPattern: p[d * s1 + t * s2])) } }
        } else {
            let p = enc.dataPointer.bindMemory(to: Float.self, capacity: capacity)
            for d in 0..<dm { for t in 0..<tFull { floats[d * tFull + t] = p[d * s1 + t * s2] } }
        }
        return MLXArray(floats, [1, dm, tFull])
    }
}
#endif
