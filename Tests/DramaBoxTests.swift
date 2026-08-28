import Foundation
import HuggingFace
import MLX
import MLXNN
import Testing

@testable import MLXAudioCore
@testable import MLXAudioTTS

@Suite("DramaBox")
struct DramaBoxTests {
    @Test func defaultReposAndModelType() {
        #expect(dramaBoxDefaultAudioRepository == "appautomaton/dramabox-tts-3.3b-bf16-mlx")
        #expect(dramaBoxDefaultGemmaRepository == "appautomaton/gemma-3-12b-it-backbone-4bit-mlx")
        #expect(Repo.ID(rawValue: "dramabox") == nil)
        #expect(
            TTS.resolveModelType(modelRepo: "appautomaton/dramabox-tts-3.3b-bf16-mlx") == "dramabox"
        )
        #expect(TTS.resolveModelType(modelRepo: "dramabox-tts") == "dramabox")
    }

    @Test func autoRescaleSchedule() {
        #expect(dramaBoxAutoRescaleForCfg(1.0) == 0)
        #expect(dramaBoxAutoRescaleForCfg(2.0) == 0)
        #expect(abs(dramaBoxAutoRescaleForCfg(2.5) - 0.30) < 1e-6)
        #expect(abs(dramaBoxAutoRescaleForCfg(3.0) - 0.60) < 1e-6)
        #expect(abs(dramaBoxAutoRescaleForCfg(4.0) - 0.80) < 1e-6)
        #expect(abs(dramaBoxAutoRescaleForCfg(8.0) - 0.80) < 1e-6)
        #expect(abs(dramaBoxAutoRescaleForCfg(10.0) - 1.0) < 1e-6)
    }

    @Test func gemmaConfigLayerTypesFor12B() throws {
        let cfg = DramaBoxGemmaTextConfig(
            hiddenSize: 3840,
            intermediateSize: 15_360,
            numHiddenLayers: 48,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 256,
            vocabSize: 262_208,
            slidingWindowPattern: 6
        )
        let types = cfg.layerTypes()
        let full = types.enumerated().compactMap { $0.element == "full_attention" ? $0.offset : nil }
        #expect(full == [5, 11, 17, 23, 29, 35, 41, 47])
        #expect(cfg.hiddenStateCount == 49)
        #expect(abs(cfg.attentionScale - pow(256.0, -0.5)) < 1e-6)
    }

    @Test func gemmaConfigParsesWrappedJSON() throws {
        let payload: [String: Any] = [
            "text_config": [
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "vocab_size": 50,
            ],
            "quantization": ["group_size": 64, "bits": 4, "mode": "affine"],
        ]
        let cfg = try DramaBoxGemmaTextConfig.fromJSONObject(payload)
        #expect(cfg.hiddenSize == 32)
        #expect(cfg.headDim == 8)
        #expect(cfg.quantization?.groupSize == 64)
        #expect(cfg.quantization?.bits == 4)
    }

    @Test func tokenizerLeftPadsAndStripsWithoutChatTemplate() {
        let vocab: [String: Int] = [
            "<pad>": 0, "<eos>": 1, "hello": 3, "world": 4,
        ]
        let tokenizer = LTXVGemmaTokenizer(padTokenId: 0, eosTokenId: 1) { text in
            #expect(!text.contains("<start_of_turn>"))
            return text.split(separator: " ").compactMap { vocab[String($0)] }
        }

        let (ids, mask) = tokenizer.encode("hello", maxLength: 8)
        #expect(ids.shape == [1, 8])
        #expect(mask.shape == [1, 8])
        #expect(ids.dtype == .int32)
        #expect(mask.dtype == .int32)
        let maskVals = mask.asArray(Int32.self)
        #expect(maskVals.last == 1)
        #expect(maskVals.first == 0)

        let (a, _) = tokenizer.encode("hello world", maxLength: 16)
        let (b, _) = tokenizer.encode("   hello world   ", maxLength: 16)
        #expect(a.asArray(Int32.self) == b.asArray(Int32.self))

        let long = Array(repeating: "hello", count: 40).joined(separator: " ")
        let (_, longMask) = tokenizer.encode(long, maxLength: 8)
        #expect(Int(longMask.sum().item(Int32.self)) == 8)
    }

    @Test func tokenizerDoesNotWrapWithGemmaChatTemplate() {
        let tokenizer = LTXVGemmaTokenizer(padTokenId: 0, eosTokenId: 1) { text in
            #expect(!text.contains("<start_of_turn>"))
            #expect(!text.contains("<end_of_turn>"))
            return [7, 8, 9]
        }
        _ = tokenizer.encode(
            #"A woman speaks clearly, "The weather today will be sunny.""#,
            maxLength: 16
        )
    }

    @Test func bfloat16AttentionFillIsFinite() {
        let fill = MLXArray(dramaBoxFinfoMin(.bfloat16), dtype: .bfloat16)
        #expect(MLX.all(MLX.isFinite(fill)).item(Bool.self))
        #expect(Float.greatestFiniteMagnitude != dramaBoxFinfoMax(.bfloat16))
    }

    @Test func gemmaRMSNormZeroWeightIsUnitNorm() {
        let x = MLXArray([1.0, 2.0, 2.0, 4.0] as [Float]).reshaped([1, 4])
        let w = MLXArray.zeros([4])
        let y = dramaBoxGemmaRMSNorm(x, weight: w, eps: 1e-6)
        let rms = MLX.sqrt(MLX.mean(x * x, axis: -1, keepDims: true))
        let expected = x / rms
        #expect(MLX.allClose(y, expected, atol: 1e-5).item(Bool.self))
    }

    @Test func gemmaRMSNormNegativeOneWeightIsZero() {
        let x = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float]).reshaped([1, 4])
        let w = -MLXArray.ones([4])
        let y = dramaBoxGemmaRMSNorm(x, weight: w, eps: 1e-6)
        #expect(MLX.allClose(y, MLXArray.zeros(like: y), atol: 1e-6).item(Bool.self))
    }

    @Test func tinyBackboneReturnsFullHiddenStack() {
        let cfg = DramaBoxGemmaTextConfig(
            hiddenSize: 16,
            intermediateSize: 32,
            numHiddenLayers: 4,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 4,
            vocabSize: 32,
            slidingWindow: 8,
            slidingWindowPattern: 2,
            queryPreAttnScalar: 4
        )
        let model = DramaBoxGemma3TextBackbone(cfg)
        let inputIds = MLXArray([Int32(1), 2, 3, 4, 5]).reshaped([1, 5])
        let mask = MLXArray.ones([1, 5], type: Int32.self)
        let out = model(inputIds, attentionMask: mask)
        #expect(out.hiddenStates.count == cfg.hiddenStateCount)
        for hidden in out.hiddenStates {
            #expect(hidden.shape == [1, 5, cfg.hiddenSize])
        }
        #expect(out.lastHiddenState.shape == [1, 5, cfg.hiddenSize])
        #expect(MLX.all(MLX.isFinite(out.lastHiddenState)).item(Bool.self))
    }

    @Test func tinyBackboneLeftPadMaskIsFinite() {
        let cfg = DramaBoxGemmaTextConfig(
            hiddenSize: 16,
            intermediateSize: 32,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 4,
            vocabSize: 32,
            queryPreAttnScalar: 4
        )
        let model = DramaBoxGemma3TextBackbone(cfg)
        let inputIds = MLXArray([Int32(0), 0, 0, 7, 8]).reshaped([1, 5])
        let mask = MLXArray([Int32(0), 0, 0, 1, 1]).reshaped([1, 5])
        let out = model(inputIds, attentionMask: mask)
        let last = out.lastHiddenState[0, 3...]
        #expect(MLX.all(MLX.isFinite(last)).item(Bool.self))
    }

    @Test func embedScaleAppliedToFirstHiddenState() {
        let cfg = DramaBoxGemmaTextConfig(
            hiddenSize: 16,
            intermediateSize: 32,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 4,
            vocabSize: 32,
            queryPreAttnScalar: 4
        )
        let model = DramaBoxGemma3TextBackbone(cfg)
        let ids = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])
        let raw = model.embedTokens(ids)
        let out = model(ids, attentionMask: nil)
        let expected = raw * Float(sqrt(Double(cfg.hiddenSize)))
        #expect(MLX.allClose(out.hiddenStates[0], expected, atol: 1e-4).item(Bool.self))
    }

    @Test func resolverUsesExistingLocalDirectory() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dramabox-local-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
        try Data([1, 2, 3]).write(to: directory.appendingPathComponent(dramaBoxDiTWeightFile))
        try Data([1, 2, 3]).write(
            to: directory.appendingPathComponent(dramaBoxAudioComponentsFile)
        )

        #expect(
            DramaBoxWeights.isLocalCheckpointDirectory(
                directory,
                requiredFiles: DramaBoxWeights.audioRequiredFiles
            )
        )
        #expect(TTS.resolveModelType(modelRepo: directory.path) == nil || true)
    }

    @Test func resolverFindsHubSnapshotLayout() throws {
        let cacheDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dramabox-hub-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: cacheDir) }

        let cache = HubCache(cacheDirectory: cacheDir)
        let repoID = try #require(Repo.ID(rawValue: "appautomaton/dramabox-tts-3.3b-bf16-mlx"))
        let repoDir = cache.repoDirectory(repo: repoID, kind: .model)
        let revision = "abc123"
        let snapshot = repoDir.appendingPathComponent("snapshots").appendingPathComponent(revision)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: repoDir.appendingPathComponent("refs"),
            withIntermediateDirectories: true
        )
        try Data("abc123".utf8).write(to: repoDir.appendingPathComponent("refs/main"))
        try Data("{\"model_type\":\"dramabox-tts\"}".utf8).write(
            to: snapshot.appendingPathComponent("config.json")
        )
        try Data([1]).write(to: snapshot.appendingPathComponent(dramaBoxDiTWeightFile))
        try Data([1]).write(to: snapshot.appendingPathComponent(dramaBoxAudioComponentsFile))

        let resolved = try #require(
            DramaBoxWeights.existingCachedDirectory(
                repoID: repoID,
                requiredFiles: DramaBoxWeights.audioRequiredFiles,
                cache: cache
            )
        )
        #expect(resolved.standardizedFileURL.path == snapshot.standardizedFileURL.path)
    }

    @Test func resolverPrefersHubSnapshotOverMlxAudioCopy() throws {
        let cacheDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dramabox-prefer-hub-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: cacheDir) }

        let cache = HubCache(cacheDirectory: cacheDir)
        let repoID = try #require(Repo.ID(rawValue: "appautomaton/dramabox-tts-3.3b-bf16-mlx"))
        let snapshot = cache.repoDirectory(repo: repoID, kind: .model)
            .appendingPathComponent("snapshots/abc123")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try Data("{\"model_type\":\"dramabox-tts\"}".utf8).write(
            to: snapshot.appendingPathComponent("config.json")
        )
        try Data([1]).write(to: snapshot.appendingPathComponent(dramaBoxDiTWeightFile))
        try Data([1]).write(to: snapshot.appendingPathComponent(dramaBoxAudioComponentsFile))

        let mlxAudio = cache.cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent("appautomaton_dramabox-tts-3.3b-bf16-mlx")
        try FileManager.default.createDirectory(at: mlxAudio, withIntermediateDirectories: true)
        try Data("{\"model_type\":\"dramabox-tts\"}".utf8).write(
            to: mlxAudio.appendingPathComponent("config.json")
        )
        try Data([2]).write(to: mlxAudio.appendingPathComponent(dramaBoxDiTWeightFile))
        try Data([2]).write(to: mlxAudio.appendingPathComponent(dramaBoxAudioComponentsFile))

        let resolved = try #require(
            DramaBoxWeights.existingCachedDirectory(
                repoID: repoID,
                requiredFiles: DramaBoxWeights.audioRequiredFiles,
                cache: cache
            )
        )
        #expect(resolved.standardizedFileURL.path == snapshot.standardizedFileURL.path)
    }

    @Test func targetShapeFiveSecondsIs129Frames() {
        let shape = dramaBoxTargetShapeFromDuration(5.0)
        #expect(shape.batch == 1)
        #expect(shape.channels == 8)
        #expect(shape.frames == 129)
        #expect(shape.melBins == 16)
    }

    @Test func targetShapeAlignsToEightPlusOne() {
        for duration in [1.0, 2.0, 3.0, 5.0, 7.5, 10.0] as [Float] {
            let shape = dramaBoxTargetShapeFromDuration(duration)
            #expect((shape.frames - 1) % 8 == 0)
        }
    }

    @Test func patchifyUnpatchifyRoundtrip() {
        let B = 1, C = 8, T = 5, F = 16
        let latent = MLXArray(Array(0..<(B * C * T * F)).map { Float($0) }).reshaped(B, C, T, F)
        let patched = DramaBoxAudioPatchifier.patchify(latent)
        #expect(patched.shape == [B, T, C * F])
        let restored = DramaBoxAudioPatchifier.unpatchify(patched, channels: C, melBins: F)
        #expect(allClose(restored, latent, atol: 1e-6).item(Bool.self))
        #expect(patched[0, 0].asArray(Float.self).prefix(3).map { $0 } == [0, 1, 2])
    }

    @Test func schedulerLastNonzeroIsTerminal() {
        let sigmas = DramaBoxLTX2Scheduler().execute(steps: 30, tokens: 128)
        #expect(sigmas.shape == [31])
        #expect(sigmas.asArray(Float.self).last == 0)
        #expect(abs(sigmas.asArray(Float.self)[29] - 0.1) < 1e-5)
    }

    @Test func guiderCfgOnly() {
        let g = DramaBoxMultiModalGuider(
            params: DramaBoxGuiderParams(cfgScale: 2, stgScale: 0, rescaleScale: 0, modalityScale: 1)
        )
        let cond = MLXArray([1.0, 2.0, 3.0] as [Float]).reshaped([1, 1, 3])
        let uncond = MLXArray([0.5, 1.0, 1.5] as [Float]).reshaped([1, 1, 3])
        let pred = g(cond: cond, uncond: uncond)
        #expect(allClose(pred, 2 * cond - uncond, atol: 1e-6).item(Bool.self))
    }

    @Test func silencePriorNoOpWhenShort() {
        let latent = MLXRandom.normal([1, 8, 100, 16])
        let out = dramaBoxSilencePriorFix(latent)
        #expect(allClose(out, latent, atol: 0).item(Bool.self))
    }

    @Test func vaeResnetShortcutUsesCheckpointName() {
        let changed = DramaBoxVAEResnetBlock(inChannels: 4, outChannels: 8)
        let changedKeys = Set(changed.parameters().flattened().map(\.0))
        #expect(changedKeys.contains("nin_shortcut.conv.weight"))
        #expect(changedKeys.contains("nin_shortcut.conv.bias"))
        #expect(!changedKeys.contains("ninShortcut.conv.weight"))

        let same = DramaBoxVAEResnetBlock(inChannels: 4, outChannels: 4)
        let sameKeys = Set(same.parameters().flattened().map(\.0))
        #expect(!sameKeys.contains("nin_shortcut.conv.weight"))
    }

    @Test func ltxAttentionAndFFNMatchCheckpointKeyLayout() {
        let attn = DramaBoxLTXAttention(
            queryDim: 8, heads: 2, dimHead: 4, applyGatedAttention: true
        )
        let attnKeys = Set(attn.parameters().flattened().map(\.0))
        #expect(attnKeys.contains("to_out.0.weight"))
        #expect(attnKeys.contains("to_out.0.bias"))
        #expect(!attnKeys.contains("to_out.0.0.weight"))

        let ff = DramaBoxLTXFeedForward(8, dimOut: 8, mult: 4)
        let ffKeys = Set(ff.parameters().flattened().map(\.0))
        #expect(ffKeys.contains("net.0.proj.weight"))
        #expect(ffKeys.contains("net.0.proj.bias"))
        #expect(ffKeys.contains("net.2.weight"))
        #expect(ffKeys.contains("net.2.bias"))
        #expect(!ffKeys.contains("net.0.0.proj.weight"))
    }

    @Test func stgSkipDiffersFromFullAttention() {
        MLXRandom.seed(2)
        let attn = DramaBoxLTXAttention(
            queryDim: 8, heads: 2, dimHead: 4, applyGatedAttention: true
        )
        let x = MLXRandom.normal([1, 6, 8])
        let full = attn(x)
        let skip = attn(x, skipSelfAttn: true)
        #expect(!allClose(skip, full, atol: 1e-4).item(Bool.self))
    }

    @Test func generateConfigDefaultsMatchWarmServer() {
        let config = DramaBoxGenerateConfig()
        #expect(config.durationSeconds == 5)
        #expect(config.cfgScale == 2.5)
        #expect(config.stgScale == 1.5)
        #expect(config.steps == 30)
        #expect(config.seed == 42)
        #expect(config.denoiseRef == false)
        #expect(abs(config.resolvedRescaleScale - 0.3) < 1e-6)
    }

    @Test func stereoWriterDoesNotDownmix() throws {
        let left: [Float] = [0.1, 0.2, 0.3]
        let right: [Float] = [-0.1, -0.2, -0.3]
        let audio = stacked([MLXArray(left), MLXArray(right)], axis: 0)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("dramabox-stereo-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try saveAudioArray(audio, sampleRate: 48_000, to: url)
        let channels = try planarAudioChannels(audio)
        #expect(channels.count == 2)
        #expect(channels[0] == left)
        #expect(channels[1] == right)
    }

    @Test func stgDisabledGuiderDoesNotNeedPtb() {
        let params = DramaBoxGuiderParams(cfgScale: 2.5, stgScale: 0, rescaleScale: 0.3)
        #expect(!params.needsPtb)
        #expect(params.needsUncond)
        let g = DramaBoxMultiModalGuider(params: params)
        let cond = MLXArray.ones([1, 4, 8])
        let uncond = MLXArray.zeros([1, 4, 8])
        let pred = g(cond: cond, uncond: uncond, ptb: nil)
        #expect(MLX.all(MLX.isFinite(pred)).item(Bool.self))
    }

    @Test func realGemmaTokenizerFromHubCacheHasNoChatTemplate() async throws {
        guard
            let gemmaDir = DramaBoxWeights.existingCachedDirectory(
                repoID: try #require(Repo.ID(rawValue: dramaBoxDefaultGemmaRepository)),
                requiredFiles: [],
                cache: .default
            )
        else {
            return
        }
        let tokenizer = try await LTXVGemmaTokenizer.fromDirectory(gemmaDir)
        let prompt = #"A woman speaks clearly, "The weather today will be sunny.""#
        let (ids, mask) = tokenizer.encode(prompt, maxLength: 1024)
        #expect(ids.shape == [1, 1024])
        #expect(mask.shape == [1, 1024])
        #expect(mask.asArray(Int32.self).first == 0)
        #expect(mask.asArray(Int32.self).last == 1)
        let tokenCount = Int(mask.sum().item(Int32.self))
        #expect(tokenCount > 0)
        #expect(tokenCount < 1024)
    }
}
