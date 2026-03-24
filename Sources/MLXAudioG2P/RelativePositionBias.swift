import MLX
import MLXNN

func relativePositionBucket(
    relativePosition: MLXArray,
    bidirectional: Bool,
    numBuckets: Int,
    maxDistance: Int
) -> MLXArray {
    var numBuckets = numBuckets
    var relativePosition = relativePosition
    var relativeBuckets = MLXArray.zeros(like: relativePosition)

    if bidirectional {
        numBuckets /= 2
        relativeBuckets = relativeBuckets + (relativePosition .> 0).asType(.int32) * numBuckets
        relativePosition = abs(relativePosition)
    } else {
        relativePosition = -minimum(relativePosition, MLXArray.zeros(like: relativePosition))
    }

    let maxExact = numBuckets / 2
    let isSmall = relativePosition .< maxExact

    let relativePositionFloat = relativePosition.asType(.float32)
    let maxExactF = MLXArray(Float(maxExact))
    let maxDistF = MLXArray(Float(maxDistance))
    var relativePositionIfLarge = maxExactF + (
        log(relativePositionFloat / maxExactF)
            / log(maxDistF / maxExactF)
            * Float(numBuckets - maxExact)
    )
    relativePositionIfLarge = minimum(
        relativePositionIfLarge,
        MLXArray(Float(numBuckets - 1))
    )

    relativeBuckets = relativeBuckets + `where`(
        isSmall,
        relativePosition,
        relativePositionIfLarge.asType(.int32)
    )
    return relativeBuckets
}

public class RelativePositionBias: Module {
    let bidirectional: Bool
    let numBuckets: Int
    let maxDistance: Int
    @ModuleInfo var embeddings: Embedding

    public init(
        numHeads: Int,
        numBuckets: Int = 32,
        maxDistance: Int = 128,
        bidirectional: Bool = true
    ) {
        self.bidirectional = bidirectional
        self.numBuckets = numBuckets
        self.maxDistance = maxDistance
        self._embeddings.wrappedValue = Embedding(
            embeddingCount: numBuckets, dimensions: numHeads
        )
    }

    public func callAsFunction(
        queryLength: Int, keyLength: Int, offset: Int = 0
    ) -> MLXArray {
        let contextPosition = MLXArray(
            Int32(offset) ..< Int32(offset + queryLength)
        ).expandedDimensions(axis: 1)
        let memoryPosition = MLXArray(
            Int32(0) ..< Int32(keyLength)
        ).expandedDimensions(axis: 0)
        let relativePosition = memoryPosition - contextPosition

        let buckets = relativePositionBucket(
            relativePosition: relativePosition,
            bidirectional: bidirectional,
            numBuckets: numBuckets,
            maxDistance: maxDistance
        )
        return embeddings(buckets).transposed(2, 0, 1)
    }
}
