import Testing
import MLX

@testable import MLXAudioSTT

struct IncrementalMelSpectrogramTests {
    @Test func firstChunkSingleSampleDoesNotCrash() {
        let mel = IncrementalMelSpectrogram(sampleRate: 16000, nFft: 400, hopLength: 160, nMels: 128)
        let out = mel.process(samples: [0.1])
        #expect(out == nil)
    }

    @Test func firstChunkTwoSamplesDoesNotCrash() {
        let mel = IncrementalMelSpectrogram(sampleRate: 16000, nFft: 400, hopLength: 160, nMels: 128)
        let out = mel.process(samples: [0.1, -0.2])
        #expect(out == nil)
    }
}

