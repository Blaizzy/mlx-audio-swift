import MLX
import Testing

@testable import MLXAudioSTT

struct VoxtralRealtimeDtypeTests {
    private static func weights(conv: DType) -> [String: MLXArray] {
        [
            "encoder.conv_layers_0_conv.conv.weight": MLXArray.zeros([8, 3, 4], dtype: conv),
            "encoder.conv_layers_0_conv.conv.bias": MLXArray.zeros([8], dtype: .float32),
            "encoder.transformer_layers.0.attention_norm.weight": MLXArray.ones([8], dtype: .float32),
            "decoder.layers.0.ffn_norm.weight": MLXArray.ones([8], dtype: .float32),
            "decoder.layers.0.ada_rms_norm_t_cond.ada_down.weight": MLXArray.zeros([4, 8], dtype: .float32),
            "decoder.norm.weight": MLXArray.ones([8], dtype: .float32),
        ]
    }

    @Test(arguments: [DType.float16, .bfloat16])
    func float32NormsFollowTheCheckpointDtype(dtype: DType) {
        let sanitized = VoxtralRealtimeModel.sanitize(weights: Self.weights(conv: dtype))

        #expect(sanitized.count == 6)
        #expect(sanitized.values.allSatisfy { $0.dtype == dtype })
    }

    @Test func float32CheckpointStaysFloat32() {
        let sanitized = VoxtralRealtimeModel.sanitize(weights: Self.weights(conv: .float32))

        #expect(sanitized.values.allSatisfy { $0.dtype == .float32 })
    }
}
