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

