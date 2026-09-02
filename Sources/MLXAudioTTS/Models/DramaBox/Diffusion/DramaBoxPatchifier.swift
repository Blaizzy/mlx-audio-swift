import Foundation
@preconcurrency import MLX

struct DramaBoxAudioLatentShape: Sendable {
    var batch: Int
    var channels: Int
    var frames: Int
    var melBins: Int

    func toTuple() -> [Int] { [batch, channels, frames, melBins] }
    func tokenCount() -> Int { frames }

    init(batch: Int, channels: Int, frames: Int, melBins: Int) {
        self.batch = batch
        self.channels = channels
        self.frames = frames
        self.melBins = melBins
    }

    init(_ target: DramaBoxAudioTargetShape) {
        self.batch = target.batch
        self.channels = target.channels
        self.frames = target.frames
        self.melBins = target.melBins
    }
}

struct DramaBoxAudioPatchifier {
    var sampleRate: Int
    var hopLength: Int
    var audioLatentDownsampleFactor: Int
    var isCausal: Bool
    var shift: Int

    init(
        sampleRate: Int = 16_000,
        hopLength: Int = 160,
        audioLatentDownsampleFactor: Int = 4,
        isCausal: Bool = true,
        shift: Int = 0
    ) {
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.audioLatentDownsampleFactor = audioLatentDownsampleFactor
        self.isCausal = isCausal
        self.shift = shift
    }

    static func patchify(_ latent: MLXArray) -> MLXArray {
        let B = latent.dim(0)
        let C = latent.dim(1)
        let T = latent.dim(2)
        let F = latent.dim(3)
        return latent.transposed(0, 2, 1, 3).reshaped(B, T, C * F)
    }

    static func unpatchify(_ latent: MLXArray, channels: Int, melBins: Int) -> MLXArray {
        let B = latent.dim(0)
        let T = latent.dim(1)
        return latent.reshaped(B, T, channels, melBins).transposed(0, 2, 1, 3)
    }

    func latentTimeInSec(start: Int, end: Int) -> MLXArray {
        let idx = MLXArray(Array(start..<end).map { Float($0) })
        var melFrame = idx * Float(audioLatentDownsampleFactor)
        if isCausal {
            melFrame = MLX.maximum(
                melFrame + (1.0 - Float(audioLatentDownsampleFactor)),
                MLXArray(0 as Float)
            )
        }
        return melFrame * Float(hopLength) / Float(sampleRate)
    }

    func getPatchGridBounds(_ shape: DramaBoxAudioLatentShape) -> MLXArray {
        var start = latentTimeInSec(start: shift, end: shape.frames + shift)
        var end = latentTimeInSec(start: shift + 1, end: shape.frames + shift + 1)
        start = MLX.broadcast(start.reshaped([1, 1, shape.frames]), to: [shape.batch, 1, shape.frames])
        end = MLX.broadcast(end.reshaped([1, 1, shape.frames]), to: [shape.batch, 1, shape.frames])
        return stacked([start, end], axis: -1)
    }
}
