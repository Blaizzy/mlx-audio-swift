import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXAudioSTT
import MLXAudioCore

@Suite(.serialized)
struct HiggsAudio3Tests {
    private func tinyConfig() -> HiggsAudio3Config {
        var config = HiggsAudio3Config()
        config.audioEncoderConfig.numMelBins = 4
        config.audioEncoderConfig.encoderLayers = 1
        config.audioEncoderConfig.encoderAttentionHeads = 2
        config.audioEncoderConfig.encoderFfnDim = 16
        config.audioEncoderConfig.dModel = 8
        config.audioEncoderConfig.maxSourcePositions = 32
        config.textConfig = Qwen3TextConfig(
            vocabSize: 32, hiddenSize: 8, intermediateSize: 16, numHiddenLayers: 1,
            numAttentionHeads: 2, numKeyValueHeads: 1, headDim: 4,
            maxPositionEmbeddings: 128, tieWordEmbeddings: false)
        config.vadCut = false
        config.chunkSizeSeconds = 0.1
        return config
    }

    private func tinyModel() throws -> HiggsAudio3Model {
        let model = HiggsAudio3Model(tinyConfig())
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { key, value in
            let phase = Float(key.utf8.reduce(0) { $0 + Int($1) } % 31) / 10
            var weight = sin(arange(value.size, dtype: .float32) * 0.13 + phase).reshaped(value.shape) * 0.05
            if key.contains("norm"), key.hasSuffix("weight") { weight = weight + 1 }
            return (key, weight)
        })
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.train(false)
        eval(model)
        return model
    }

    @Test func audioFeaturesAndCachedDecoder() throws {
        let model = try tinyModel()
        let waveform = sin(arange(1600, dtype: .float32) * 0.03)
        let features = model.audioFeatures(HiggsAudio3Frontend.features(waveform, numMelBins: 4))
        #expect(features.dim(1) > 0)
        #expect(features.shape.last == model.config.textConfig.hiddenSize)
        #expect(MLX.all(isFinite(features)).item(Bool.self))

        let ids = MLXArray([Int32(3), 5, 8, 9]).reshaped(1, 4)
        let full = model(inputIds: ids)
        eval(full)
        let cache: [KVCache] = [KVCacheSimple()]
        eval(model(inputIds: ids[0..., 0..<3], cache: cache))
        let cached = model(inputIds: ids[0..., 3..<4], cache: cache)
        #expect(cached.shape == [1, 1, model.config.textConfig.vocabSize])
        let maximumError = abs(cached - full[0..., 3..<4, 0...]).max().item(Float.self)
        #expect(maximumError < 2e-4)
    }

    @Test func chunkBoundariesPreserveSilenceAndSplitLongRegions() {
        #expect(HiggsAudio3Model.chunkRanges(sampleCount: 100, maxSamples: 25) == [0..<25,25..<50,50..<75,75..<100])
        #expect(HiggsAudio3Model.chunkRanges(sampleCount: 100, maxSamples: 50,
            speechRanges: [(10,20),(70,80)]) == [0..<20,20..<70,70..<100])
        #expect(HiggsAudio3Model.chunkRanges(sampleCount: 100, maxSamples: 50,
            speechRanges: [(10,20),(70,80)], splitVADs: true) == [10..<20,70..<80])
        #expect(HiggsAudio3Model.chunkRanges(sampleCount: 0, maxSamples: 50).isEmpty)
    }
    @Test func outputParsingHandlesMultilineThinkingAndControlTokens() {
        #expect(HiggsAudio3Model.parseOutput("<think>one\ntwo</think>hello<|im_end|>") == "hello")
        #expect(HiggsAudio3Model.parseOutput("<think>unfinished\nhello") == "unfinished\nhello")
    }
    @Test func sanitizerPreservesMLXLayoutsAndIndependentHead() {
        let raw: [String:MLXArray] = [
            "audio_tower.conv1.weight": MLXArray.zeros([8,4,3]),
            "audio_encoder_proj.temporal.weight": MLXArray.zeros([8,1,3]),
            "audio_decoder_proj.text_lm_head.weight": MLXArray.zeros([32,8]),
            "audio_decoder_proj.audio_lm_head.weight": MLXArray.zeros([1,8]),
            "layers.0.input_layernorm.weight": MLXArray.ones([8]),
            "embed_tokens.weight": MLXArray.zeros([32,8])]
        let prepared = HiggsAudio3Model.sanitize(weights: raw)
        #expect(prepared["audio_tower.conv1.weight"]?.shape == [8,3,4])
        #expect(prepared["audio_encoder_proj.temporal.weight"]?.shape == [8,3,1])
        #expect(prepared["lm_head.weight"]?.shape == [32,8])
        #expect(prepared["audio_decoder_proj.audio_lm_head.weight"] == nil)
        #expect(prepared["model.layers.0.input_layernorm.weight"] != nil)
        let again = HiggsAudio3Model.sanitize(weights: prepared)
        #expect(again["audio_tower.conv1.weight"]?.shape == [8,3,4])
    }
    @Test func shortAudioHasNonemptyPooledFeatures() throws {
        let model = try tinyModel()
        let mel = HiggsAudio3Frontend.features(MLXArray([Float(0.1)]), numMelBins: 4)
        #expect(model.audioFeatures(mel).dim(1) > 0)
    }
    @Test func missingTokenizerReturnsActionableError() throws {
        let model = try tinyModel()
        #expect(throws: STTError.self) { try model.transcribe(audio: MLXArray.ones([1600])) }
    }
}

/// Opt-in real weights: HIGGS_AUDIO3_MODEL_DIR must name an already downloaded snapshot.
@Suite(.serialized)
struct HiggsAudio3NetworkTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["HIGGS_AUDIO3_MODEL_DIR"] != nil))
    func realCheckpointTranscription() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["HIGGS_AUDIO3_MODEL_DIR"])
        let model = try await HiggsAudio3Model.fromPretrained(path, loadVAD: false)
        let url = try #require(Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media"))
        let (_,audio) = try loadAudioArray(from: url, sampleRate: 16000)
        let result = try model.transcribe(audio: audio,
            generationParameters: STTGenerateParameters(maxTokens: 256, topP: 1, chunkDuration: 4))
        print("HIGGS_REAL_TRANSCRIPT: \(result.text)")
        print("HIGGS_REAL_METRICS: prompt=\(result.promptTokens) generated=\(result.generationTokens) time=\(result.totalTime) peakGB=\(result.peakMemoryUsage)")
        #expect(!result.text.isEmpty)
        #expect(result.generationTokens > 0)
        let conversationURL = try #require(Bundle.module.url(forResource: "conversational_a", withExtension: "wav", subdirectory: "media"))
        let (_,conversation) = try loadAudioArray(from: conversationURL, sampleRate: 16000)
        let multiword = try model.transcribe(audio: conversation,
            generationParameters: STTGenerateParameters(maxTokens: 256, topP: 1, chunkDuration: 4))
        print("HIGGS_CONVERSATION_TRANSCRIPT: \(multiword.text)")
        print("HIGGS_CONVERSATION_METRICS: prompt=\(multiword.promptTokens) generated=\(multiword.generationTokens) time=\(multiword.totalTime)")
        #expect(multiword.text.split(separator: " ").count > 3)
    }
}
