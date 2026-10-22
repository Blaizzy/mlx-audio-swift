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

    private func fixture() throws -> (HiggsAudio3Model, [String:MLXArray]) {
        let weightsURL = try #require(Bundle.module.url(forResource: "higgs_audio3_reference", withExtension: "safetensors", subdirectory: "media"))
        let config = tinyConfig()
        let arrays = try MLX.loadArrays(url: weightsURL)
        let model = HiggsAudio3Model(config)
        let weights = Dictionary(uniqueKeysWithValues: arrays.compactMap { key,value in
            key.hasPrefix("weights.") ? (String(key.dropFirst(8)),value) : nil
        })
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.train(false)
        return (model,arrays)
    }
    private func compare(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float = 2e-4) {
        #expect(actual.shape == expected.shape)
        let maximum = abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
        print("Higgs reference max absolute error \(maximum), tolerance \(tolerance)")
        #expect(maximum < tolerance)
    }
    @Test func pythonFrontendEncoderProjectorAndCachedDecoder() throws {
        let (model,f) = try fixture()
        compare(HiggsAudio3Frontend.features(f["reference.wave"]!, numMelBins: 4), f["reference.frontend"]!)
        let encoded = model.audioTower(f["reference.mel"]!)
        compare(encoded, f["reference.encoder"]!)
        compare(model.projector(encoded), f["reference.projector"]!)
        let cache: [KVCache] = [KVCacheSimple()]
        compare(model(inputIds: MLXArray([Int32(3),5,8]).reshaped(1,3), cache: cache), f["reference.logits"]!)
        compare(model(inputIds: MLXArray([Int32(9)]).reshaped(1,1), cache: cache), f["reference.cached_logits"]!)
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
        let (model,_) = try fixture()
        let mel = HiggsAudio3Frontend.features(MLXArray([Float(0.1)]), numMelBins: 4)
        #expect(model.audioFeatures(mel).dim(1) > 0)
    }
    @Test func missingTokenizerReturnsActionableError() throws {
        let (model,_) = try fixture()
        #expect(throws: STTError.self) { try model.transcribe(audio: MLXArray.ones([1600])) }
    }
}

/// Opt-in real weights: HIGGS_AUDIO3_MODEL_DIR must name an already downloaded snapshot.
@Suite(.serialized)
struct HiggsAudio3NetworkTests {
    @Test func realCheckpointTranscription() async throws {
        guard let path = ProcessInfo.processInfo.environment["HIGGS_AUDIO3_MODEL_DIR"] else { return }
        let model = try await HiggsAudio3Model.fromPretrained(path, loadVAD: false)
        let url = try #require(Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media"))
        let (_,audio) = try loadAudioArray(from: url, sampleRate: 16000)
        if let referencePath = ProcessInfo.processInfo.environment["HIGGS_AUDIO3_REFERENCE"] {
            let f = try MLX.loadArrays(url: URL(fileURLWithPath: referencePath))
            let mel = HiggsAudio3Frontend.features(f["audio"]!).asType(model.audioTower.conv1.weight.dtype)
            let encoded = model.audioTower(mel)
            let projected = model.audioFeatures(mel)
            let inputs = try model.inputEmbeddings(audio: f["audio"]!)
            let cache: [KVCache] = (0..<model.config.textConfig.numHiddenLayers).map { _ in KVCacheSimple() }
            let hidden = model.textModel(inputsEmbeds: inputs.embeddings, cache: cache)
            let logits = model.lmHead(hidden[0..., -1, 0...].expandedDimensions(axis: 1))
            for (name,actual,expected) in [
                ("mel",mel,f["mel"]!), ("encoder",encoded,f["encoded"]!),
                ("projector",projected,f["projected"]!), ("embeddings",inputs.embeddings,f["embeddings"]!),
                ("logits",logits,f["logits"]!)
            ] {
                let a = actual.asType(.float32).reshaped([-1]), e = expected.asType(.float32).reshaped([-1])
                #expect(actual.shape == expected.shape)
                let difference = abs(a-e).max().item(Float.self)
                let rms = sqrt(mean((a-e).square())).item(Float.self)
                let cosine = (sum(a*e) / sqrt(sum(a.square()) * sum(e.square()))).item(Float.self)
                print("HIGGS_REAL_PARITY \(name): max_abs=\(difference) rms=\(rms) cosine=\(cosine)")
                #expect(cosine > 0.995)
            }
            #expect(inputs.ids == f["ids"]!.asArray(Int32.self).map(Int.init))
            #expect(logits[0,-1,0...].argMax().item(Int.self) == f["logits"]![0,-1,0...].argMax().item(Int.self))
        }
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
