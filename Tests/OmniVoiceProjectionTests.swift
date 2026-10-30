import Foundation
import Testing
import MLX
import MLXNN
@testable import MLXAudioTTS

@Suite("OmniVoice target projection parity", .serialized)
struct OmniVoiceProjectionTests {
    @Test(arguments: [false, true])
    func targetProjectionMatchesFullOutput(quantized: Bool) throws {
        if let path = ProcessInfo.processInfo.environment["OMNIVOICE_TEST_METALLIB"] {
            GPU.metallib = URL(fileURLWithPath: path)
        }
        let json = #"{"audio_vocab_size":17,"audio_mask_id":16,"num_audio_codebook":2,"llm_config":{"hidden_size":64,"intermediate_size":128,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"vocab_size":64}}"#
        let config = try JSONDecoder().decode(OmniVoiceConfig.self, from: Data(json.utf8))
        let model = try OmniVoiceModel(config: config)
        if quantized { quantize(model: model, groupSize: 64, bits: 8) }
        let ids = MLXArray.zeros([1, 2, 12], dtype: .int32)
        let mask = MLXArray([false, false, false, false, true, true, true, true, true, true, true, true]).reshaped([1, 12])
        let full = model.forward(inputIds: ids, audioMask: mask)
        let target = model.forward(inputIds: ids, audioMask: mask, targetStart: 4)
        #expect(target.shape == [1, 8, 2, 17])
        let difference = abs(full[0..., 4..., 0..., 0...] - target).max().item(Float.self)
        #expect(difference < 0.00001)
    }
}
