import Foundation

struct DramaBoxAudioTargetShape: Sendable {
    var batch: Int
    var channels: Int
    var frames: Int
    var melBins: Int

    func toTuple() -> [Int] { [batch, channels, frames, melBins] }
}

func dramaBoxTargetShapeFromDuration(
    _ durationS: Float,
    batch: Int = 1,
    fps: Float = 25,
    channels: Int = 8,
    melBins: Int = 16,
    sampleRate: Int = 16_000,
    hopLength: Int = 160,
    audioLatentDownsampleFactor: Int = 4
) -> DramaBoxAudioTargetShape {
    var nFrames = Int((durationS * fps).rounded()) + 1
    nFrames = ((nFrames - 1 + 4) / 8) * 8 + 1
    let latentsPerSecond = Float(sampleRate) / Float(hopLength) / Float(audioLatentDownsampleFactor)
    let audioFrames = Int((Float(nFrames) / fps * latentsPerSecond).rounded())
    return DramaBoxAudioTargetShape(
        batch: batch,
        channels: channels,
        frames: audioFrames,
        melBins: melBins
    )
}
