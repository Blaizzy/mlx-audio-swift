import Testing
@testable import MLXAudioG2P

struct MLXAudioG2PSmokeTests {
    @Test func moduleImports() {
        #expect(MLXAudioG2P.version == "0.1.0")
    }

    @Test func pipelineReturnsStructuredOutput() throws {
        let pipeline = G2PPipeline.preview()
        let output = try pipeline.convert("Hello world")
        #expect(!output.tokens.isEmpty)
        #expect(!output.phonemes.isEmpty)
        #expect(!output.normalizedText.isEmpty)
    }

    @Test func pipelineRejectsEmptyInput() {
        let pipeline = G2PPipeline.preview()
        #expect(throws: G2PError.emptyInput) {
            try pipeline.convert("")
        }
    }

    @Test func pipelineRejectsWhitespaceOnly() {
        let pipeline = G2PPipeline.preview()
        #expect(throws: G2PError.emptyInput) {
            try pipeline.convert("   \n  ")
        }
    }

    @Test func pipelineFallsBackForUnknownWord() throws {
        let pipeline = G2PPipeline.preview()
        let output = try pipeline.convert("zorb")
        #expect(!output.phonemes.isEmpty)
    }

    @Test func phonemeRenderIsStableForSameInput() throws {
        let pipeline = G2PPipeline.preview()
        let output1 = try pipeline.convert("Hello world")
        let output2 = try pipeline.convert("Hello world")
        #expect(output1.phonemes.render() == output2.phonemes.render())
    }
}
