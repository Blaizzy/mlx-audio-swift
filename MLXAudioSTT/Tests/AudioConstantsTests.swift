import Testing
@testable import MLXAudioSTT

struct AudioConstantsTests {
    @Test func sampleRateIs16kHz() {
        #expect(AudioConstants.sampleRate == 16000)
    }

    @Test func nFFTIsWhisperWindowSize() {
        #expect(AudioConstants.nFFT == 400)
    }

    @Test func hopLengthIs160() {
        #expect(AudioConstants.hopLength == 160)
    }

    @Test func nMelsIs80() {
        #expect(AudioConstants.nMels == 80)
    }

    @Test func nFramesIs3000() {
        #expect(AudioConstants.nFrames == 3000)
    }

    @Test func chunkLengthIs30Seconds() {
        #expect(AudioConstants.chunkLength == 30)
    }

    @Test func nSamplesMatchesChunkLength() {
        // 30 seconds * 16000 samples/second = 480000 samples
        #expect(AudioConstants.nSamples == 480000)
    }
}
