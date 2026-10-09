import MLX
import Testing

@testable import MLXAudioTTS

@Suite("Chatterbox generation tests")
struct ChatterboxGenerationTests {
    @Test func emotionOverrideAppliesToDefaultConditioning() {
        let defaults = MLXArray(Float(0.5))

        #expect(
            resolveChatterboxEmotionAdv(default: defaults, override: nil).item(Float.self) == 0.5)
        #expect(
            resolveChatterboxEmotionAdv(default: defaults, override: 0.9).item(Float.self) == 0.9)
    }
}
