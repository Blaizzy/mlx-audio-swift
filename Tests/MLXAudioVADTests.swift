//
//  MLXAudioVADTests.swift
//  MLXAudioTests
//
//  Created by Prince Canuma on 10/02/2026.
//

import Foundation
import Testing
import MLX
import MLXNN

@testable import MLXAudioCore
@testable import MLXAudioVAD


// MARK: - Configuration Tests

struct SortformerConfigTests {

    @Test func fcEncoderConfigDefaults() throws {
        let json = "{}"
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(FCEncoderConfig.self, from: data)

        #expect(config.hiddenSize == 512)
        #expect(config.numHiddenLayers == 18)
        #expect(config.numAttentionHeads == 8)
        #expect(config.numKeyValueHeads == 8)
        #expect(config.intermediateSize == 2048)
        #expect(config.numMelBins == 80)
        #expect(config.convKernelSize == 9)
        #expect(config.subsamplingFactor == 8)
        #expect(config.subsamplingConvChannels == 256)
        #expect(config.subsamplingConvKernelSize == 3)
        #expect(config.subsamplingConvStride == 2)
        #expect(config.maxPositionEmbeddings == 5000)
        #expect(config.attentionBias == true)
        #expect(config.scaleInput == true)
    }

    @Test func fcEncoderConfigCustom() throws {
        let json = """
        {
            "hidden_size": 256,
            "num_hidden_layers": 6,
            "num_attention_heads": 4,
            "intermediate_size": 1024,
            "num_mel_bins": 40,
            "scale_input": false
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(FCEncoderConfig.self, from: data)

        #expect(config.hiddenSize == 256)
        #expect(config.numHiddenLayers == 6)
        #expect(config.numAttentionHeads == 4)
        #expect(config.intermediateSize == 1024)
        #expect(config.numMelBins == 40)
        #expect(config.scaleInput == false)
    }

    @Test func tfEncoderConfigDefaults() throws {
        let json = "{}"
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(TFEncoderConfig.self, from: data)

        #expect(config.dModel == 192)
        #expect(config.encoderLayers == 18)
        #expect(config.encoderAttentionHeads == 8)
        #expect(config.encoderFfnDim == 768)
        #expect(config.layerNormEps == 1e-5)
        #expect(config.maxSourcePositions == 1500)
        #expect(config.kProjBias == false)
    }

    @Test func tfEncoderConfigCustom() throws {
        let json = """
        {
            "d_model": 128,
            "encoder_layers": 6,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 512,
            "k_proj_bias": true
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(TFEncoderConfig.self, from: data)

        #expect(config.dModel == 128)
        #expect(config.encoderLayers == 6)
        #expect(config.encoderAttentionHeads == 4)
        #expect(config.encoderFfnDim == 512)
        #expect(config.kProjBias == true)
    }

    @Test func modulesConfigDefaults() throws {
        let json = "{}"
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(ModulesConfig.self, from: data)

        #expect(config.numSpeakers == 4)
        #expect(config.fcDModel == 512)
        #expect(config.tfDModel == 192)
        #expect(config.subsamplingFactor == 8)
        #expect(config.chunkLen == 188)
        #expect(config.fifoLen == 0)
        #expect(config.spkcacheLen == 188)
        #expect(config.useAosc == false)
        #expect(config.silThreshold == 0.1)
    }

    @Test func processorConfigDefaults() throws {
        let json = "{}"
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(ProcessorConfig.self, from: data)

        #expect(config.featureSize == 80)
        #expect(config.samplingRate == 16000)
        #expect(config.hopLength == 160)
        #expect(config.nFft == 512)
        #expect(config.winLength == 400)
        #expect(config.preemphasis == 0.97)
    }

    @Test func sortformerConfigDecoding() throws {
        let json = """
        {
            "model_type": "sortformer",
            "num_speakers": 4,
            "fc_encoder_config": {
                "hidden_size": 512,
                "num_hidden_layers": 18,
                "num_mel_bins": 80
            },
            "tf_encoder_config": {
                "d_model": 192,
                "encoder_layers": 18
            },
            "modules_config": {
                "num_speakers": 4,
                "fc_d_model": 512,
                "tf_d_model": 192
            },
            "processor_config": {
                "sampling_rate": 16000,
                "hop_length": 160
            }
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(SortformerConfig.self, from: data)

        #expect(config.modelType == "sortformer")
        #expect(config.numSpeakers == 4)
        #expect(config.fcEncoderConfig.hiddenSize == 512)
        #expect(config.fcEncoderConfig.numHiddenLayers == 18)
        #expect(config.tfEncoderConfig.dModel == 192)
        #expect(config.tfEncoderConfig.encoderLayers == 18)
        #expect(config.modulesConfig.numSpeakers == 4)
        #expect(config.processorConfig.samplingRate == 16000)
    }

    @Test func sortformerConfigAllDefaults() throws {
        let json = "{}"
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(SortformerConfig.self, from: data)

        #expect(config.modelType == "sortformer")
        #expect(config.numSpeakers == 4)
        #expect(config.fcEncoderConfig.hiddenSize == 512)
        #expect(config.tfEncoderConfig.dModel == 192)
        #expect(config.modulesConfig.fcDModel == 512)
        #expect(config.processorConfig.featureSize == 80)
    }
}

// MARK: - VADOutput Tests

struct VADOutputTests {

    @Test func diarizationSegmentCreation() {
        let segment = DiarizationSegment(start: 1.5, end: 3.0, speaker: 0)

        #expect(segment.start == 1.5)
        #expect(segment.end == 3.0)
        #expect(segment.speaker == 0)
    }

    @Test func diarizationOutputRTTMText() {
        let segments = [
            DiarizationSegment(start: 0.0, end: 1.0, speaker: 0),
            DiarizationSegment(start: 1.5, end: 2.5, speaker: 1),
        ]

        let output = DiarizationOutput(segments: segments, numSpeakers: 2)
        let text = output.text

        #expect(text.contains("speaker_0"))
        #expect(text.contains("speaker_1"))
        #expect(text.contains("SPEAKER audio 1"))
    }

    @Test func diarizationOutputEmpty() {
        let output = DiarizationOutput(segments: [])
        #expect(output.text == "")
        #expect(output.numSpeakers == 0)
    }

    @Test func streamingStateInit() {
        let embDim = 512
        let nSpk = 4
        let state = StreamingState(
            spkcache: MLXArray.zeros([1, 0, embDim]),
            spkcachePreds: MLXArray.zeros([1, 0, nSpk]),
            fifo: MLXArray.zeros([1, 0, embDim]),
            fifoPreds: MLXArray.zeros([1, 0, nSpk]),
            framesProcessed: 0,
            meanSilEmb: MLXArray.zeros([1, embDim]),
            nSilFrames: MLXArray.zeros([1])
        )

        #expect(state.spkcacheLen == 0)
        #expect(state.fifoLen == 0)
        #expect(state.framesProcessed == 0)
    }
}

// MARK: - Feature Extraction Tests

struct SortformerFeatureTests {

    @Test func preemphasisFilterShape() {
        let waveform = MLXArray.ones([16000])
        let filtered = preemphasisFilter(waveform)

        #expect(filtered.shape == waveform.shape)
    }

    @Test func preemphasisFilterFirstSample() {
        let waveform = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float])
        let filtered = preemphasisFilter(waveform, coeff: 0.97)
        eval(filtered)

        // First sample should be unchanged
        let first = filtered[0].item(Float.self)
        #expect(first == 1.0)

        // Second sample: 2.0 - 0.97 * 1.0 = 1.03
        let second = filtered[1].item(Float.self)
        #expect(abs(second - 1.03) < 1e-4)
    }

    @Test func extractMelFeaturesShape() {
        // 1 second of audio at 16kHz
        let waveform = MLXRandom.normal([16000])
        let features = extractMelFeatures(waveform)
        eval(features)

        // Should be (batch=1, nMels=80, numFrames) with numFrames padded to multiple of 16
        #expect(features.ndim == 3)
        #expect(features.dim(0) == 1)
        #expect(features.dim(1) == 80)
        #expect(features.dim(2) % 16 == 0)
        #expect(features.dim(2) > 0)
    }

    @Test func extractMelFeaturesNoPad() {
        let waveform = MLXRandom.normal([16000])
        let features = extractMelFeatures(waveform, padTo: 0)
        eval(features)

        #expect(features.ndim == 3)
        #expect(features.dim(0) == 1)
        #expect(features.dim(1) == 80)
        #expect(features.dim(2) > 0)
    }

    @Test func extractMelFeaturesBatched() {
        let waveform = MLXRandom.normal([2, 16000])
        let features = extractMelFeatures(waveform)
        eval(features)

        #expect(features.ndim == 3)
        #expect(features.dim(0) == 2)
        #expect(features.dim(1) == 80)
    }

    @Test func trimSilenceNoTrim() {
        // All-speech waveform (high energy)
        let waveform = MLXRandom.normal([16000]) * 0.5
        let (trimmed, _) = trimSilence(waveform, sampleRate: 16000)
        eval(trimmed)

        #expect(trimmed.dim(0) > 0)
        // May or may not trim depending on random values, just verify it runs
    }

    @Test func trimSilenceShortAudio() {
        // Very short audio should not be trimmed
        let waveform = MLXRandom.normal([1000])
        let (trimmed, offset) = trimSilence(waveform, sampleRate: 16000)
        eval(trimmed)

        #expect(trimmed.dim(0) == 1000)
        #expect(offset == 0)
    }
}

// MARK: - Weight Sanitization Tests

struct SortformerSanitizeTests {

    @Test func sanitizeConv2dWeights() {
        // Simulate PyTorch Conv2d weights: (O, I, H, W)
        let weights: [String: MLXArray] = [
            "fc_encoder.subsampling.layers.0.weight": MLXArray.ones([256, 1, 3, 3]),
        ]

        let sanitized = SortformerModel.sanitize(weights)

        // Should rename layers.0 → layers_0 and transpose to (O, H, W, I)
        let w = sanitized["fc_encoder.subsampling.layers_0.weight"]!
        #expect(w.shape == [256, 3, 3, 1])
    }

    @Test func sanitizeConv1dWeights() {
        // Simulate PyTorch Conv1d weights: (O, I, K)
        let weights: [String: MLXArray] = [
            "fc_encoder.layers.0.conv.pointwise_conv1.weight": MLXArray.ones([1024, 512, 1]),
        ]

        let sanitized = SortformerModel.sanitize(weights)

        // Should transpose to (O, K, I)
        let w = sanitized["fc_encoder.layers.0.conv.pointwise_conv1.weight"]!
        #expect(w.shape == [1024, 1, 512])
    }

    @Test func sanitizeSkipsNumBatchesTracked() {
        let weights: [String: MLXArray] = [
            "fc_encoder.layers.0.conv.norm.num_batches_tracked": MLXArray([0]),
            "fc_encoder.layers.0.conv.norm.weight": MLXArray.ones([512]),
        ]

        let sanitized = SortformerModel.sanitize(weights)

        #expect(sanitized["fc_encoder.layers.0.conv.norm.num_batches_tracked"] == nil)
        #expect(sanitized["fc_encoder.layers.0.conv.norm.weight"] != nil)
    }

    @Test func sanitizeAlreadyConvertedPassesThrough() {
        // When weights already use layers_ format, skip conversion
        let weights: [String: MLXArray] = [
            "fc_encoder.subsampling.layers_0.weight": MLXArray.ones([256, 3, 3, 1]),
        ]

        let sanitized = SortformerModel.sanitize(weights)

        let w = sanitized["fc_encoder.subsampling.layers_0.weight"]!
        // Should NOT transpose again
        #expect(w.shape == [256, 3, 3, 1])
    }

    @Test func sanitizeDepthwiseConvWeights() {
        let weights: [String: MLXArray] = [
            "fc_encoder.layers.0.conv.depthwise_conv.weight": MLXArray.ones([512, 1, 9]),
        ]

        let sanitized = SortformerModel.sanitize(weights)

        let w = sanitized["fc_encoder.layers.0.conv.depthwise_conv.weight"]!
        #expect(w.shape == [512, 9, 1])
    }
}

// MARK: - Post-Processing Tests

struct SortformerPostprocessingTests {

    @Test func predsToSegmentsBasic() {
        // Create simple predictions: speaker 0 active for frames 0-9, speaker 1 for frames 5-14
        let nFrames = 20
        let nSpk = 2
        var predsData = [Float](repeating: 0.0, count: nFrames * nSpk)

        // Speaker 0: frames 0-9
        for i in 0..<10 { predsData[i * nSpk + 0] = 0.8 }
        // Speaker 1: frames 5-14
        for i in 5..<15 { predsData[i * nSpk + 1] = 0.9 }

        let preds = MLXArray(predsData).reshaped(nFrames, nSpk)
        let frameDuration: Float = 0.08  // 80ms per frame

        let segments = SortformerModel.predsToSegments(preds, frameDuration: frameDuration)

        #expect(segments.count >= 2)

        let speakers = Set(segments.map { $0.speaker })
        #expect(speakers.contains(0))
        #expect(speakers.contains(1))

        // All segments should have positive duration
        for seg in segments {
            #expect(seg.end > seg.start)
        }
    }

    @Test func predsToSegmentsEmpty() {
        // All predictions below threshold
        let preds = MLXArray.zeros([20, 4])
        let segments = SortformerModel.predsToSegments(preds, frameDuration: 0.08)

        #expect(segments.isEmpty)
    }

    @Test func predsToSegmentsWithMinDuration() {
        // Create a very short active region (2 frames = 0.16s)
        let nFrames = 20
        let nSpk = 2
        var predsData = [Float](repeating: 0.0, count: nFrames * nSpk)
        predsData[5 * nSpk + 0] = 0.9
        predsData[6 * nSpk + 0] = 0.9

        let preds = MLXArray(predsData).reshaped(nFrames, nSpk)

        // With minDuration = 0.5, the short segment should be filtered out
        let segments = SortformerModel.predsToSegments(
            preds, frameDuration: 0.08, minDuration: 0.5
        )
        #expect(segments.isEmpty)

        // Without minDuration, it should appear
        let segmentsNoMin = SortformerModel.predsToSegments(
            preds, frameDuration: 0.08, minDuration: 0.0
        )
        #expect(segmentsNoMin.count == 1)
    }

    @Test func predsToSegmentsWithMergeGap() {
        // Two close segments that should be merged
        let nFrames = 30
        let nSpk = 1
        var predsData = [Float](repeating: 0.0, count: nFrames * nSpk)

        // Segment 1: frames 0-4
        for i in 0..<5 { predsData[i] = 0.9 }
        // Gap: frames 5-6 (0.16s)
        // Segment 2: frames 7-14
        for i in 7..<15 { predsData[i] = 0.9 }

        let preds = MLXArray(predsData).reshaped(nFrames, nSpk)

        // Without merge: should have 2 segments
        let segmentsNoMerge = SortformerModel.predsToSegments(
            preds, frameDuration: 0.08, mergeGap: 0.0
        )
        #expect(segmentsNoMerge.count == 2)

        // With mergeGap = 0.5s: should merge into 1
        let segmentsMerged = SortformerModel.predsToSegments(
            preds, frameDuration: 0.08, mergeGap: 0.5
        )
        #expect(segmentsMerged.count == 1)
    }

    @Test func predsToSegmentsSorted() {
        // Multiple speakers — output should be sorted by start time
        let nFrames = 20
        let nSpk = 3
        var predsData = [Float](repeating: 0.0, count: nFrames * nSpk)

        // Speaker 2: early (frames 0-4)
        for i in 0..<5 { predsData[i * nSpk + 2] = 0.9 }
        // Speaker 0: middle (frames 8-12)
        for i in 8..<13 { predsData[i * nSpk + 0] = 0.9 }
        // Speaker 1: late (frames 15-19)
        for i in 15..<20 { predsData[i * nSpk + 1] = 0.9 }

        let preds = MLXArray(predsData).reshaped(nFrames, nSpk)
        let segments = SortformerModel.predsToSegments(preds, frameDuration: 0.08)

        // Should be sorted by start time
        for i in 1..<segments.count {
            #expect(segments[i].start >= segments[i - 1].start)
        }
    }
}

// MARK: - Smart Turn Config Tests

struct SmartTurnConfigTests {

    @Test func smartTurnConfigDefaults() throws {
        let json = "{}"
        let data = json.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(SmartTurnConfig.self, from: data)

        #expect(cfg.modelType == "smart_turn")
        #expect(cfg.architecture == "smart_turn")
        #expect(cfg.dtype == "float32")
        #expect(cfg.encoderConfig.numMelBins == 80)
        #expect(cfg.processorConfig.samplingRate == 16000)
        #expect(cfg.processorConfig.maxAudioSeconds == 8)
    }

    @Test func smartTurnConfigFromDict() throws {
        let json = """
        {
            "dtype": "float16",
            "sample_rate": 22050,
            "max_audio_seconds": 6,
            "threshold": 0.42,
            "encoder_config": {
                "num_mel_bins": 8,
                "max_source_positions": 64,
                "d_model": 16,
                "encoder_attention_heads": 2,
                "encoder_layers": 1,
                "encoder_ffn_dim": 32,
                "k_proj_bias": false
            },
            "processor_config": {
                "sampling_rate": 16000,
                "max_audio_seconds": 8,
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 8,
                "normalize_audio": true,
                "threshold": 0.5
            }
        }
        """
        let data = json.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(SmartTurnConfig.self, from: data)

        #expect(cfg.dtype == "float16")
        #expect(cfg.sampleRate == 22050)
        #expect(cfg.maxAudioSeconds == 6)
        #expect(abs(cfg.threshold - 0.42) < 1e-6)
        #expect(cfg.encoderConfig.dModel == 16)
        #expect(cfg.processorConfig.nMels == 8)
    }

    @Test func smartTurnSynthesizesProcessorConfig() throws {
        let json = """
        {
            "sample_rate": 24000,
            "max_audio_seconds": 5,
            "threshold": 0.33,
            "encoder_config": { "num_mel_bins": 64 }
        }
        """
        let data = json.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(SmartTurnConfig.self, from: data)

        #expect(cfg.processorConfig.samplingRate == 24000)
        #expect(cfg.processorConfig.maxAudioSeconds == 5)
        #expect(cfg.processorConfig.nMels == 64)
        #expect(abs(cfg.processorConfig.threshold - 0.33) < 1e-6)
    }
}

// MARK: - Smart Turn Model Tests

private func makeTinySmartTurnConfig(dtype: String = "float32") -> SmartTurnConfig {
    let encoder = SmartTurnEncoderConfig(
        numMelBins: 8,
        maxSourcePositions: 64,
        dModel: 16,
        encoderAttentionHeads: 2,
        encoderLayers: 1,
        encoderFfnDim: 32,
        kProjBias: false
    )
    let processor = SmartTurnProcessorConfig(
        samplingRate: 16000,
        maxAudioSeconds: 8,
        nFft: 400,
        hopLength: 160,
        nMels: 8,
        normalizeAudio: true,
        threshold: 0.5
    )
    return SmartTurnConfig(dtype: dtype, encoderConfig: encoder, processorConfig: processor)
}

private func makeTinySmartTurnModel(dtype: String = "float32") throws -> SmartTurnModel {
    let model = SmartTurnModel(makeTinySmartTurnConfig(dtype: dtype))
    eval(model.parameters())

    if dtype == "float16" {
        let casted = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { key, value in
                (key, value.asType(.float16))
            }
        )
        try model.update(parameters: ModuleParameters.unflattened(casted), verify: .noUnusedKeys)
        eval(model.parameters())
    }

    return model
}

struct SmartTurnForwardTests {

    @Test func smartTurnForwardShapeAndRange() throws {
        let model = try makeTinySmartTurnModel()
        let input = MLXArray.zeros([1, 8, 64], type: Float.self)
        let out = model(input)
        eval(out)

        #expect(out.shape == [1, 1])
        let minVal = out.min().item(Float.self)
        let maxVal = out.max().item(Float.self)
        #expect(minVal >= 0.0)
        #expect(maxVal <= 1.0)
    }

    @Test func smartTurnForwardReturnLogits() throws {
        let model = try makeTinySmartTurnModel()
        let input = MLXArray.zeros([1, 8, 64], type: Float.self)
        let logits = model(input, returnLogits: true)
        eval(logits)

        #expect(logits.shape == [1, 1])
    }

    @Test func smartTurnForwardBatchDimension() throws {
        let model = try makeTinySmartTurnModel()
        let input = MLXArray.zeros([2, 8, 64], type: Float.self)
        let out = model(input)
        eval(out)

        #expect(out.shape == [2, 1])
    }

    @Test func smartTurnDTypePropagation() throws {
        let fp32Model = try makeTinySmartTurnModel(dtype: "float32")
        let fp32In = MLXArray.zeros([1, 8, 64], type: Float.self)
        let fp32Out = fp32Model(fp32In)
        eval(fp32Out)
        #expect(fp32Model.modelDType == .float32)
        #expect(fp32Out.dtype == .float32)

        let fp16Model = try makeTinySmartTurnModel(dtype: "float16")
        let fp16In = MLXArray.zeros([1, 8, 64], type: Float.self).asType(.float16)
        let fp16Out = fp16Model(fp16In)
        eval(fp16Out)
        #expect(fp16Model.modelDType == .float16)
        #expect(fp16Out.dtype == .float16)
    }

    @Test func smartTurnPrepareAudioArrayLengths() throws {
        let model = try makeTinySmartTurnModel()
        let maxSamples = model.config.processorConfig.maxAudioSeconds * model.config.processorConfig.samplingRate

        let short = MLXArray.ones([16000], type: Float.self)
        let shortOut = try model.prepareAudioSamples(short, sampleRate: 16000)
        #expect(shortOut.count == maxSamples)

        let long = MLXArray.ones([200000], type: Float.self)
        let longOut = try model.prepareAudioSamples(long, sampleRate: 16000)
        #expect(longOut.count == maxSamples)
    }

    @Test func smartTurnPrepareAudioArrayResample() throws {
        let model = try makeTinySmartTurnModel()
        let maxSamples = model.config.processorConfig.maxAudioSeconds * model.config.processorConfig.samplingRate

        let audio8k = MLXArray.ones([8000], type: Float.self)
        let out = try model.prepareAudioSamples(audio8k, sampleRate: 8000)
        #expect(out.count == maxSamples)
    }

    @Test func smartTurnPrepareInputFeaturesShape() throws {
        let model = try makeTinySmartTurnModel()
        let audio = MLXArray.zeros([16000], type: Float.self)
        let features = try model.prepareInputFeatures(audio, sampleRate: 16000)
        eval(features)

        #expect(features.shape == [8, 800])
    }

    @Test func smartTurnPredictEndpointReturnsOutput() throws {
        let model = try makeTinySmartTurnModel()
        let audio = MLXArray.zeros([16000], type: Float.self)
        let result = try model.predictEndpoint(audio, sampleRate: 16000, threshold: 0.5)

        #expect(result.prediction == 0 || result.prediction == 1)
        #expect(result.probability >= 0.0 && result.probability <= 1.0)
    }
}

// MARK: - Smart Turn Sanitization Tests

struct SmartTurnSanitizeTests {

    @Test func smartTurnSanitizeDropsValConstants() {
        let sanitized = SmartTurnModel.sanitize([
            "val_17": MLXArray.zeros([16, 16], type: Float.self),
            "val_123": MLXArray.zeros([1], type: Float.self)
        ])
        #expect(sanitized.isEmpty)
    }

    @Test func smartTurnSanitizeRemapsPrefixes() {
        let sanitized = SmartTurnModel.sanitize([
            "inner.classifier.0.weight": MLXArray.zeros([16, 16], type: Float.self),
            "inner.pool_attention.2.bias": MLXArray.zeros([1], type: Float.self)
        ])
        #expect(sanitized["classifier_0.weight"] != nil)
        #expect(sanitized["pool_attention_2.bias"] != nil)
    }

    @Test func smartTurnSanitizeConv1dTranspose() {
        let weights: [String: MLXArray] = [
            "encoder.conv1.weight": MLXArray.zeros([16, 8, 3], type: Float.self)
        ]
        let sanitized = SmartTurnModel.sanitize(weights)
        #expect(sanitized["encoder.conv1.weight"]?.shape == [16, 3, 8])
    }

    @Test func smartTurnSanitizeFCTransposeHeuristics() {
        let weights: [String: MLXArray] = [
            "encoder.layers.0.fc1.weight": MLXArray.zeros([16, 32], type: Float.self),
            "encoder.layers.0.fc2.weight": MLXArray.zeros([32, 16], type: Float.self),
        ]
        let sanitized = SmartTurnModel.sanitize(weights)
        #expect(sanitized["encoder.layers.0.fc1.weight"]?.shape == [32, 16])
        #expect(sanitized["encoder.layers.0.fc2.weight"]?.shape == [16, 32])
    }

    @Test func smartTurnSanitizePoolTransposeHeuristics() {
        let weights: [String: MLXArray] = [
            "pool_attention.0.weight": MLXArray.zeros([16, 256], type: Float.self),
            "pool_attention.2.weight": MLXArray.zeros([256, 1], type: Float.self),
        ]
        let sanitized = SmartTurnModel.sanitize(weights)
        #expect(sanitized["pool_attention_0.weight"]?.shape == [256, 16])
        #expect(sanitized["pool_attention_2.weight"]?.shape == [1, 256])
    }
}

// MARK: - Smart Turn Network Tests

struct SmartTurnNetworkTests {

    @Test func smartTurnFromPretrainedEvaluatesConversationalAudio() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network SmartTurn test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_SMARTTURN_REPO"] ?? "mlx-community/smart-turn-v3"
        let model = try await SmartTurnModel.fromPretrained(repo)

        let audioURLTrue = Bundle.module.url(
            forResource: "conversational_a",
            withExtension: "wav",
            subdirectory: "media"
        )!
        let (_, audioTrue) = try loadAudioArray(from: audioURLTrue, sampleRate: 16000)
        let resultTrue = try model.predictEndpoint(audioTrue, sampleRate: 16000, threshold: 0.5)
        #expect(resultTrue.prediction == 1)
        #expect(resultTrue.probability >= 0.5 && resultTrue.probability <= 1.0)

        let audioURLFalse = Bundle.module.url(
            forResource: "false-turn",
            withExtension: "wav",
            subdirectory: "media"
        )!
        let (_, audioFalse) = try loadAudioArray(from: audioURLFalse, sampleRate: 16000)
        let resultFalse = try model.predictEndpoint(audioFalse, sampleRate: 16000, threshold: 0.5)
        #expect(resultFalse.prediction == 0)
        #expect(resultFalse.probability >= 0.0 && resultFalse.probability < 0.5)
    }
}
