import Foundation
import MLX
import MLXNN
import MLXAudioCore
import Testing
@testable import MLXAudioTTS

@Suite(.serialized)
struct VoxtralTTSTests {
    func config() -> VoxtralTTSConfiguration {
        var c = VoxtralTTSConfiguration()
        c.dim = 8; c.layers = 1; c.headDim = 4; c.hiddenDim = 16; c.heads = 2; c.kvHeads = 1
        c.vocabularySize = 2048; c.semanticSize = 4; c.acousticSize = 5; c.acousticCodebooks = 2
        c.acousticDim = 8; c.acousticLayers = 1; c.acousticHeadDim = 4; c.acousticHiddenDim = 16
        c.acousticHeads = 2; c.acousticKVHeads = 1
        c.tokenizerDim = 8; c.tokenizerHiddenDim = 16; c.tokenizerHeads = 2; c.tokenizerKVHeads = 1
        c.tokenizerHeadDim = 4; c.patchSize = 3; c.patchKernel = 3; c.semanticDim = 2; c.tokenizerAcousticDim = 2
        c.decoderLengths = [1,1]; c.decoderKernels = [3,4]; c.decoderStrides = [1,2]
        return c
    }
    func fixture() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        let special = [("<s>",1),("[AUDIO]",24),("[BEGIN_AUDIO]",25),("[REPEAT_AUDIO_TEXT]",35),("[NEXT_AUDIO_TEXT]",36)]
        let json: [String: Any] = ["config": ["default_num_special_tokens": 1000, "default_vocab_size": 1256],
            "vocab": (0..<256).map { ["token_bytes": Data([UInt8($0)]).base64EncodedString()] },
            "special_tokens": special.map { ["token_str": $0.0, "rank": $0.1] as [String: Any] },
            "audio": ["voice_num_audio_tokens": ["test": 3]]]
        try JSONSerialization.data(withJSONObject: json).write(to: url.appendingPathComponent("tekken.json"))
        return url
    }
    func deterministic(_ module: Module) throws {
        let weights = Dictionary(uniqueKeysWithValues: module.parameters().flattened().map { key,value in
            let tensor: MLXArray
            if key.hasSuffix("norm.weight") || key.hasSuffix("_norm.weight") || key == "norm.weight" { tensor = .ones(value.shape) }
            else if key.hasSuffix("scale") { tensor = .ones(value.shape) * 0.5 }
            else if key.hasSuffix("cluster_usage") { tensor = .ones(value.shape) }
            else { tensor = ((MLXArray(0..<value.size).asType(.float32) % 7) - 3).reshaped(value.shape) / 16 }
            return (key,tensor)
        })
        try module.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    }
    @Test func configurationAndRouting() throws {
        let json = #"{"model_type":"voxtral_tts","dim":8,"n_layers":1,"multimodal":{"audio_model_args":{"semantic_codebook_size":4,"audio_encoding_args":{"sampling_rate":24000},"acoustic_transformer_args":{"n_layers":1}},"audio_tokenizer_args":{"decoder_transformer_lengths_str":"1,1","decoder_convs_kernels_str":"3,4","decoder_convs_strides_str":"1,2"}},"quantization":{"bits":4,"group_size":32,"language_model.model.model.layers.0.self_attn.q_proj":{"bits":8,"group_size":32}}}"#
        let c = try JSONDecoder().decode(VoxtralTTSConfiguration.self, from: Data(json.utf8))
        #expect(c.dim == 8 && c.layers == 1 && c.acousticLayers == 1 && c.semanticSize == 4)
        #expect(c.decoderStrides == [1,2] && c.samplesPerFrame == 480)
        #expect(c.quantization?.quantization(layer: "backbone.layers.0.attention.wq")?.bits == 8)
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit") == "voxtral_tts")
        #expect(TTS.resolveModelType(modelRepo: "local", modelType: " VOXTRAL_TTS ") == "voxtral_tts")
        var invalid = config(); invalid.acousticCodebooks = 3
        #expect(throws: AudioGenerationError.self) { try invalid.validate() }
    }
    @Test func speechFramingAndVoiceMetadata() throws {
        let directory = try fixture(); defer { try? FileManager.default.removeItem(at: directory) }
        let processor = try VoxtralTTSTextProcessor(directory: directory)
        let tokens = try processor.encode("Hi", voice: "test", sanitize: true)
        #expect(tokens == [1,25,24,24,24,36,1072,1105,1046,35,25])
        #expect(processor.tokenizer.decode(tokenIds: Array(tokens[6..<9])) == "Hi.")
        #expect(throws: AudioGenerationError.self) { try processor.encode("Hi", voice: "unknown", sanitize: true) }
        #expect(throws: AudioGenerationError.self) { try processor.encode("\u{200b}", voice: "test", sanitize: true) }
    }
    @Test func textCleanup() throws {
        #expect(try VoxtralTTSText.sanitize("# Hello **world**!!!\nRead [this](https://example.com).") == "Hello **world**! Read this.")
        #expect(try VoxtralTTSText.sanitize("Pay $1.25 and $2.05 (N/A, >= 5ms).") == "Pay one dollar and twenty-five cents and two oh five—not available, greater than or equal to 5 milliseconds—.")
        #expect(try VoxtralTTSText.sanitize("Cost €0.25 and 1,250,000 -- fine...") == "Cost twenty-five cents and one million two hundred fifty thousand — fine...")
        #expect(try VoxtralTTSText.sanitize("A &amp; B &eacute; &#x1F600;") == "A & B é 😀.")
        #expect(try VoxtralTTSText.sanitize("Bonjour le monde") == "Bonjour le monde.")
        #expect(try VoxtralTTSText.sanitize("₹१२.५०") == "twelve rupees and fifty paise.")
        #expect(VoxtralTTSText.htmlUnescape("&#x80; &#0; &#xFFFF;") == "€ � ")
    }
    @Test func weightMappingsAndNormalization() throws {
        let m = VoxtralTTSModel(config())
        let cleaned = try m.sanitize([
            "language_model.model.model.layers.0.self_attn.q_proj.scales": .ones([8,1]),
            "layers.0.feed_forward.w1.weight": .ones([16,8]),
            "audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original0": .ones([8,1,1]) * 2,
            "audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original1": .ones([8,4,3]),
            "audio_tokenizer.decoder_blocks.1.layers.0.ffn_scale": .ones([8]),
            "audio_tokenizer.quantizer.semantic_codebook.embedding_sum": .ones([4,2])])
        #expect(cleaned["backbone.layers.0.attention.wq.scales"]?.shape == [8,1])
        #expect(cleaned["backbone.layers.0.feed_forward.w1.weight"]?.shape == [16,8])
        #expect(cleaned["audio_tokenizer.stages.0.transformer.layers.0.ffn_scale"] != nil)
        #expect(cleaned["audio_tokenizer.embedding_sum"] != nil)
        #expect(abs(cleaned["audio_tokenizer.stages.0.conv.weight"]! - 2 / sqrt(Float(12))).max().item(Float.self) < 1e-6)
        #expect(cleaned.count == 5)
        let indices = m.globalIndices(MLXArray([Int32(2),2,2]).reshaped(1,3)).asArray(Int32.self)
        #expect(indices == [2,8,15])
    }
    @Test func causalBackboneCacheMatchesFullPass() throws {
        try Device.withDefaultDevice(.cpu) {
        let model = VoxtralTTSBackbone(config()); try deterministic(model)
        let input = ((MLXArray(0..<24).asType(.float32) % 5) / 8).reshaped(1,3,8)
        let full = model(input, cache: [KVCacheSimple()])
        let expected = MLXArray([Float(0.1721004992723465),0.2335619330406189,0.8417094945907593,1.327583909034729,2.087603807449341,-0.09092745184898376,0.1394357830286026,1.0286582708358765,1.3390250205993652,1.313766598701477,-0.052963823080062866,0.3323921263217926,0.9319141507148743,1.061478853225708,1.526979923248291,0.2003321796655655,0.35515666007995605,0.8496233224868774,1.2096999883651733,1.8207138776779175,-0.0318027064204216,0.45925596356391907,0.8453808426856995,1.2025456428527832]).reshaped(1,3,8)
        #expect(abs(full - expected).max().item(Float.self) < 2e-6)
        let cache: [KVCache] = [KVCacheSimple()]
        let first = model(input[0...,0..<2,0...], cache: cache)
        let last = model(input[0...,2..<3,0...], cache: cache)
        #expect(abs(full - concatenated([first,last], axis: 1)).max().item(Float.self) < 1e-5)
        }
    }
    @Test func acousticFlowAndCodec() throws {
        try Device.withDefaultDevice(.cpu) {
        let c = config(), head = VoxtralTTSAcousticHead(config()), decoder = VoxtralTTSAudioDecoder(config())
        try deterministic(head); try deterministic(decoder)
        let hidden = MLXArray([Float(0.125),-0.25,0.375,-0.5,0.125,0.25,0.5,-0.125]).reshaped(1,8)
        let noise = MLXArray([Float(-0.5),0.25]).reshaped(1,2)
        let velocity = head.velocity(noise, time: 0.5, hidden: hidden)
        #expect(abs(velocity - MLXArray([Float(-0.5130108594894409),-0.5325942635536194]).reshaped(1,2)).max().item(Float.self) < 2e-6)
        #expect(velocity.shape == [1,2] && isFinite(velocity).all().item(Bool.self))
        let codes = head.decodeFrame(hidden, options: .init(denoisingSteps: 3), noise: noise)
        #expect(codes.shape == [1,3])
        #expect(codes[0,1...].min().item(Int.self) >= 2 && codes[0,1...].max().item(Int.self) <= 6)
        let sequence = MLXArray([Int32(2),2,6,3,4,3,4,5,2]).reshaped(1,3,3)
        let waveform = decoder.decode(sequence)
        #expect(abs(waveform - MLXArray([Float(0.2536313533782959),-0.08194375783205032,-0.13228999078273773,-0.0868254005908966,-0.00462584663182497,0.17430953681468964,0.2232043296098709,-0.09595546871423721,-0.15478390455245972,-0.09549950063228607,0.0024581793695688248,0.16960227489471436,0.15146328508853912,0.0024924427270889282,-0.2638349235057831,-0.22733232378959656,0.09451161324977875,0.08066549897193909]).reshaped(1,-1)).max().item(Float.self) < 2e-6)
        #expect(abs(decoder.codebook(sequence) - MLXArray([Float(-0.1875),-0.125,-1.0,1.0,-0.0625,0.0,0.0,-0.5,0.0625,0.125,0.5,-1.0]).reshaped(1,3,4)).max().item(Float.self) == 0)
        #expect(waveform.shape == [1,3 * c.samplesPerFrame])
        #expect(isFinite(waveform).all().item(Bool.self))
        let prefix = decoder.decode(sequence[0...,0..<2,0...])
        #expect(abs(waveform[0...,0..<prefix.dim(1)] - prefix).max().item(Float.self) < 1e-6)
        }
    }
    @Test func quantizedProjectionRestoresPublishedLayout() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = VoxtralTTSAttention(dim: 64, heads: 2, kvHeads: 1, headDim: 32)
            quantize(model: source, groupSize: 32, bits: 4)
            let weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
            let restored = VoxtralTTSAttention(dim: 64, heads: 2, kvHeads: 1, headDim: 32)
            quantize(model: restored, groupSize: 32, bits: 4)
            try restored.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
            #expect(weights["wq.scales"]?.shape == [64,2])
            let input = MLXArray.ones([1,3,64]) / 8
            #expect(abs(source(input) - restored(input)).max().item(Float.self) == 0)
        }
    }
    @Test func localCheckpointVoicesAndIncrementalGeneration() async throws {
        let directory = try fixture(); defer { try? FileManager.default.removeItem(at: directory) }
        let model = VoxtralTTSModel(config())
        // Positive embeddings and one positive semantic row force audio code 2 for two frames.
        var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { key,value in
            (key, key.hasSuffix("norm.weight") || key == "audio_tokenizer.cluster_usage" ? MLXArray.ones(value.shape) : MLXArray.zeros(value.shape))
        })
        weights["backbone.embed_tokens.weight"] = .ones([2048,8])
        weights["audio_embeddings.weight"] = .ones([config().audioEmbeddingCount,8])
        var semantic = MLXArray.zeros([128,8]); semantic[2] = .ones([8])
        weights["acoustic_transformer.semantic_codebook_output.weight"] = semantic
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        let json = #"{"model_type":"voxtral_tts","dim":8,"n_layers":1,"head_dim":4,"hidden_dim":16,"n_heads":2,"n_kv_heads":1,"vocab_size":2048,"semantic_codebook_size":4,"acoustic_codebook_size":5,"n_acoustic_codebook":2,"acoustic_dim":8,"acoustic_n_layers":1,"acoustic_head_dim":4,"acoustic_hidden_dim":16,"acoustic_n_heads":2,"acoustic_n_kv_heads":1,"tokenizer_dim":8,"tokenizer_hidden_dim":16,"tokenizer_n_heads":2,"tokenizer_n_kv_heads":1,"tokenizer_head_dim":4,"tokenizer_patch_size":3,"tokenizer_patch_proj_kernel_size":3,"tokenizer_semantic_dim":2,"tokenizer_acoustic_dim":2,"tokenizer_decoder_transformer_lengths_str":"1,1","tokenizer_decoder_convs_kernels_str":"3,4","tokenizer_decoder_convs_strides_str":"1,2"}"#
        try Data(json.utf8).write(to: directory.appendingPathComponent("config.json"))
        let voiceDirectory = directory.appendingPathComponent("voice_embedding")
        try FileManager.default.createDirectory(at: voiceDirectory, withIntermediateDirectories: true)
        try MLX.save(arrays: ["embedding": MLXArray.ones([3,8]) * 0.5], url: voiceDirectory.appendingPathComponent("test.safetensors"))
        let loaded = try #require(try await TTS.loadModel(modelRepo: directory.path) as? VoxtralTTSModel)
        #expect(loaded.availableVoices == ["test"])
        #expect(loaded.acousticHead.semantic(.zeros([1,8])).item(Int.self) == 1)
        #expect(try loaded.voiceEmbedding("test").shape == [3,8])
        let ids = try loaded.promptTokens("Hi", voice: "test", options: .init())
        let embeddings = try loaded.inputEmbeddings(ids, voice: "test")
        #expect(embeddings[0,2..<5,0...].mean().item(Float.self) == 0.5)
        let audio = try await loaded.generate(text: "Hi", voice: "test", generationParameters: .init(maxTokens: 2, seed: 9))
        #expect(audio.shape == [12])
        #expect(isFinite(audio).all().item(Bool.self))
        var chunks = 0, samples = 0
        for try await event in loaded.generateStream(text: "Hi", voice: "test", refAudio: nil, refText: nil, language: nil,
            generationParameters: .init(maxTokens: 2, seed: 9), streamingInterval: 0.08) {
            if case .audio(let chunk) = event { chunks += 1; samples += chunk.size }
        }
        #expect(chunks == 2)
        #expect(samples == audio.size)
    }
    @Test func invalidGenerationFails() async throws {
        let model = VoxtralTTSModel(config())
        await #expect(throws: AudioGenerationError.self) { try await model.generate(text: "Hi", options: .init(denoisingSteps: 1)) }
        await #expect(throws: AudioGenerationError.self) { try await model.generate(text: "Hi", generationParameters: .init(maxTokens: 0)) }
        await #expect(throws: AudioGenerationError.self) {
            try await model.generate(text: "Hi", voice: nil, refAudio: .zeros([8]), refText: nil, language: nil, generationParameters: .init())
        }
    }
}
