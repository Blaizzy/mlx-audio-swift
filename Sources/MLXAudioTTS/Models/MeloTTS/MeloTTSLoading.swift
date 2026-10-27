import Foundation
import HuggingFace
import MLX
import MLXAudioCore

extension MeloTTSModel {
    static func validate(_ weights: [String: MLXArray], shapes: [String: [Int]]) throws {
        for (key, shape) in shapes {
            guard let value = weights[key], value.shape == shape, value.dtype.isFloatingPoint else {
                throw AudioGenerationError.invalidInput("Missing or incompatible MeloTTS tensor: \(key), expected \(shape)")
            }
        }
    }
    static func synthesisShapes(_ c: MeloTTSConfiguration) -> [String: [Int]] {
        var s: [String: [Int]] = [:]
        func norm(_ p: String, _ n: Int) { s[p + ".weight"] = [n]; s[p + ".bias"] = [n] }
        func linear(_ p: String, _ input: Int, _ output: Int) { s[p + ".weight"] = [output, input]; s[p + ".bias"] = [output] }
        func conv(_ p: String, _ input: Int, _ output: Int, _ kernel: Int = 1, bias: Bool = true, groups: Int = 1) {
            s[p + ".conv.weight"] = [output, kernel, input / groups]
            if bias { s[p + ".conv.bias"] = [output] }
        }
        func encoder(_ p: String, channels: Int, layers: Int, kernel: Int, speaker: Bool) {
            for i in 0..<layers {
                let a = p + ".attn_layers.\(i)"
                for v in ["q", "k", "v", "o"] { conv(a + ".conv_" + v, channels, channels) }
                s[a + ".emb_rel_k"] = [1, 9, channels / c.heads]; s[a + ".emb_rel_v"] = [1, 9, channels / c.heads]
                norm(p + ".norm_layers_1.\(i)", channels); norm(p + ".norm_layers_2.\(i)", channels)
                conv(p + ".ffn_layers.\(i).conv_1", channels, c.filterChannels, kernel)
                conv(p + ".ffn_layers.\(i).conv_2", c.filterChannels, channels, kernel)
            }
            if speaker { linear(p + ".spk_emb_linear", c.ginChannels, channels) }
        }
        func dds(_ p: String) {
            for i in 0..<3 {
                conv(p + ".convs_sep.\(i)", 192, 192, 3, groups: 192)
                conv(p + ".convs_1x1.\(i)", 192, 192)
                norm(p + ".norms_1.\(i)", 192); norm(p + ".norms_2.\(i)", 192)
            }
        }
        s["emb_g.weight"] = [c.nSpeakers, c.ginChannels]
        s["enc_p.emb.weight"] = [c.vocabularySize, c.hiddenChannels]
        s["enc_p.tone_emb.weight"] = [c.numTones, c.hiddenChannels]
        s["enc_p.language_emb.weight"] = [c.numLanguages, c.hiddenChannels]
        conv("enc_p.bert_proj", 1024, c.hiddenChannels); conv("enc_p.ja_bert_proj", 768, c.hiddenChannels)
        encoder("enc_p.encoder", channels: c.hiddenChannels, layers: c.layers, kernel: c.kernelSize, speaker: c.useSpeakerEncoder)
        conv("enc_p.proj", c.hiddenChannels, c.interChannels * 2)
        conv("dp.cond", c.ginChannels, c.hiddenChannels)
        conv("dp.conv_1", c.hiddenChannels, 256, 3); conv("dp.conv_2", 256, 256, 3)
        norm("dp.norm_1", 256); norm("dp.norm_2", 256); conv("dp.proj", 256, 1)
        conv("sdp.pre", c.hiddenChannels, 192); conv("sdp.cond", c.ginChannels, 192); conv("sdp.proj", 192, 192)
        dds("sdp.convs")
        s["sdp.flows.0.m"] = [2, 1]; s["sdp.flows.0.logs"] = [2, 1]
        for i in [1, 3, 5, 7] { let p = "sdp.flows.\(i)"; conv(p + ".pre", 1, 192); dds(p + ".convs"); conv(p + ".proj", 192, 29) }
        for i in [0, 2, 4, 6] {
            let p = "flow_layers.\(i)"
            conv(p + ".pre", c.interChannels / 2, c.interChannels)
            encoder(p + ".enc", channels: c.interChannels, layers: c.flowLayers, kernel: 5, speaker: true)
            conv(p + ".post", c.interChannels, c.interChannels / 2)
        }
        conv("dec.conv_pre", c.interChannels, c.upsampleChannels, 7); conv("dec.cond", c.ginChannels, c.upsampleChannels)
        for i in c.upsampleRates.indices {
            let channels = c.upsampleChannels / (1 << (i + 1)), p = "dec.ups.\(i).conv_t"
            s[p + ".weight"] = [channels, c.upsampleKernels[i], channels * 2]; s[p + ".bias"] = [channels]
            for j in c.resblockKernels.indices {
                for k in c.resblockDilations[j].indices {
                    for part in ["convs1", "convs2"] { conv("dec.resblocks.\(i * c.resblockKernels.count + j).\(part).\(k)", channels, channels, c.resblockKernels[j]) }
                }
            }
        }
        conv("dec.conv_post", c.upsampleChannels / (1 << c.upsampleRates.count), 1, 7, bias: false)
        return s
    }
    static func bertShapes() -> [String: [Int]] {
        var s = ["embeddings.word_embeddings.weight": [30522, 768], "embeddings.position_embeddings.weight": [512, 768], "embeddings.token_type_embeddings.weight": [2, 768]]
        for p in ["embeddings.norm"] + (0..<10).flatMap({ ["encoder.layers.\($0).ln1", "encoder.layers.\($0).ln2"] }) {
            s[p + ".weight"] = [768]; s[p + ".bias"] = [768]
        }
        for i in 0..<10 {
            for n in ["query", "key", "value", "out"] {
                let p = "encoder.layers.\(i).attention.\(n)_proj"
                s[p + ".weight"] = [768, 768]; s[p + ".bias"] = [768]
            }
            s["encoder.layers.\(i).linear1.weight"] = [3072, 768]; s["encoder.layers.\(i).linear1.bias"] = [3072]
            s["encoder.layers.\(i).linear2.weight"] = [768, 3072]; s["encoder.layers.\(i).linear2.bias"] = [768]
        }
        return s
    }
    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws -> MeloTTSModel {
        guard let repo = Repo.ID(rawValue: modelRepo) else { throw TTSModelError.invalidRepositoryID(modelRepo) }
        let directory = try await ModelUtils.resolveOrDownloadModel(repoID: repo, requiredExtension: "safetensors", additionalMatchingPatterns: ["bert_weights.npz"], cache: cache)
        return try await fromModelDirectory(directory, cache: cache)
    }
    /// `frontendDirectory` can provide the five published English frontend files for offline loading.
    public static func fromModelDirectory(_ directory: URL, cache: HubCache = .default, frontendDirectory: URL? = nil) async throws -> MeloTTSModel {
        let configuration = try JSONDecoder().decode(MeloTTSConfiguration.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        var weights: [String: MLXArray] = [:]
        for url in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil).filter({ $0.pathExtension == "safetensors" }).sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            for (key, value) in try MLX.loadArrays(url: url) {
                let key = key.replacingOccurrences(of: "flow.flows.", with: "flow_layers.")
                guard weights[key] == nil else { throw AudioGenerationError.invalidInput("Duplicate MeloTTS tensor: \(key)") }
                weights[key] = value
            }
        }
        let shapes = synthesisShapes(configuration)
        try validate(weights, shapes: shapes)
        // Posterior encoder, discriminator and training-only duration flows are not needed for synthesis.
        let synthesis = weights.filter { shapes[$0.key] != nil }
        let bertWeights = try NumpyArchive.load(url: directory.appendingPathComponent("bert_weights.npz"))
        let bShapes = bertShapes()
        try validate(bertWeights, shapes: bShapes)
        let localFrontend = directory.appendingPathComponent("frontend")
        let frontendDir: URL
        if let frontendDirectory { frontendDir = frontendDirectory }
        else if FileManager.default.fileExists(atPath: localFrontend.appendingPathComponent("vocab.txt").path) { frontendDir = localFrontend }
        else { frontendDir = try await resolveFrontend(cache: cache) }
        let frontend = try MeloEnglishFrontend(directory: frontendDir)
        return MeloTTSModel(configuration: configuration, network: MeloLayers(weights: synthesis), bert: MeloLayers(weights: bertWeights.filter { bShapes[$0.key] != nil }), frontend: frontend)
    }
    static func resolveFrontend(cache: HubCache) async throws -> URL {
        let g2pRevision = "c6439c274c42b9724a7fee1dc07ca6a4c68a0538"
        let nltkRevision = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
        let directory = cache.cacheDirectory.appendingPathComponent("mlx-audio/frontends/melotts/\(g2pRevision)-\(nltkRevision)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let sources = [
            "vocab.txt": "https://huggingface.co/google-bert/bert-base-uncased/resolve/86b5e0934494bd15c9632b12f734a8a67f723594/vocab.txt",
            "checkpoint20.npz": "https://raw.githubusercontent.com/Kyubyong/g2p/\(g2pRevision)/g2p_en/checkpoint20.npz",
            "homographs.en": "https://raw.githubusercontent.com/Kyubyong/g2p/\(g2pRevision)/g2p_en/homographs.en",
            "cmudict.zip": "https://raw.githubusercontent.com/nltk/nltk_data/\(nltkRevision)/packages/corpora/cmudict.zip",
            "averaged_perceptron_tagger_eng.zip": "https://raw.githubusercontent.com/nltk/nltk_data/\(nltkRevision)/packages/taggers/averaged_perceptron_tagger_eng.zip",
        ]
        for (name, source) in sources.sorted(by: { $0.key < $1.key }) {
            let destination = directory.appendingPathComponent(name)
            if let attrs = try? destination.resourceValues(forKeys: [.fileSizeKey]), (attrs.fileSize ?? 0) > 0 { continue }
            let (data, response) = try await URLSession.shared.data(from: URL(string: source)!)
            guard let response = response as? HTTPURLResponse, response.statusCode == 200, !data.isEmpty else {
                throw AudioGenerationError.modelNotInitialized("Could not download MeloTTS frontend \(name)")
            }
            try data.write(to: destination, options: .atomic)
        }
        return directory
    }
}
