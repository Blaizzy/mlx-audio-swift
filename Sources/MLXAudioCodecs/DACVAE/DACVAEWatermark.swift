//
//  DACVAEWatermark.swift
//  MLXAudioCodecs
//
// Created by Prince Canuma on 04/01/2026.
//

import Foundation
import MLX
import MLXNN

// MARK: - Message Processor

/// Apply the secret message to the encoder output.
public class DACVAEMsgProcessor: Module {
    let nbits: Int
    let hiddenSize: Int
    @ModuleInfo(key: "msg_processor") var msgProcessor: Embedding

    public init(nbits: Int, hiddenSize: Int) {
        self.nbits = nbits
        self.hiddenSize = hiddenSize
        self._msgProcessor.wrappedValue = Embedding(embeddingCount: 2 * nbits, dimensions: hiddenSize)
    }

    public func callAsFunction(_ hidden: MLXArray, msg: MLXArray) -> MLXArray {
        // hidden: (B, C, T) - encoder output
        // msg: (B, nbits) - binary message

        let batchSize = msg.shape[0]

        // Create indices: 0, 2, 4, ..., 2*nbits
        var offsetValues: [Int32] = []
        for i in 0..<nbits {
            offsetValues.append(Int32(i * 2))
        }
        let offsets = MLXArray(offsetValues)

        // Broadcast offsets to (B, nbits)
        let offsetsBroadcast = MLX.broadcast(offsets.expandedDimensions(axis: 0), to: [batchSize, nbits])

        // Add offsets to message
        let indices = (offsetsBroadcast + msg.asType(.int32)).asType(.int32)

        // Get embeddings: (B, nbits, hidden_size)
        let msgAux = msgProcessor(indices)

        // Sum across nbits: (B, hidden_size)
        let msgAuxSummed = msgAux.sum(axis: 1)

        // Expand and broadcast to hidden shape: (B, hidden_size, 1) -> (B, C, T)
        let msgAuxExpanded = msgAuxSummed.expandedDimensions(axis: 2)
        let msgAuxBroadcast = MLX.broadcast(msgAuxExpanded, to: hidden.shape)

        return hidden + msgAuxBroadcast
    }
}

// MARK: - Watermark Encoder Block

/// Watermark encoder block with Tanh and LSTM.
public class DACVAEWatermarkEncoderBlock: Module {
    var sharedSnakeOut: DACVAESnake1d?
    var sharedConvOut: DACVAEWNConv1d?

    @ModuleInfo(key: "pre_3") var pre3: DACVAEWNConv1d
    @ModuleInfo(key: "post_0") var post0: DACVAELSTMBlock
    @ModuleInfo(key: "post_1") var post1: DACVAEElu
    @ModuleInfo(key: "post_2") var post2: DACVAEWNConv1d

    public init(
        outDim: Int = 128,
        wmChannels: Int = 32,
        hidden: Int = 512,
        lstmLayers: Int = 2
    ) {
        // Pre-processing after shared layers: Tanh + Conv
        self._pre3.wrappedValue = DACVAEWNConv1d(
            inChannels: 1,
            outChannels: wmChannels,
            kernelSize: 7,
            causal: true,
            padMode: "auto",
            norm: "none"
        )

        // Post-processing: LSTM + ELU + Conv
        self._post0.wrappedValue = DACVAELSTMBlock(
            inputSize: hidden,
            hiddenSize: hidden,
            numLayers: lstmLayers,
            skip: true
        )
        self._post1.wrappedValue = DACVAEElu()
        self._post2.wrappedValue = DACVAEWNConv1d(
            inChannels: hidden,
            outChannels: outDim,
            kernelSize: 7,
            causal: true,
            padMode: "auto",
            norm: "none"
        )
    }

    /// Set shared layers from decoder.
    public func setSharedLayers(snakeOut: DACVAESnake1d, convOut: DACVAEWNConv1d) {
        self.sharedSnakeOut = snakeOut
        self.sharedConvOut = convOut
    }

    /// Forward through pre-processing (shared + own layers).
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let snakeOut = sharedSnakeOut, let convOut = sharedConvOut else {
            fatalError("Shared layers not set. Call setSharedLayers first.")
        }
        var h = snakeOut(x)
        h = convOut(h)
        h = MLX.tanh(h)
        h = pre3(h)
        return h
    }

    /// Forward through shared layers + tanh only (for blending).
    public func forwardNoWMConv(_ x: MLXArray) -> MLXArray {
        guard let snakeOut = sharedSnakeOut, let convOut = sharedConvOut else {
            fatalError("Shared layers not set. Call setSharedLayers first.")
        }
        var h = snakeOut(x)
        h = convOut(h)
        h = MLX.tanh(h)
        return h
    }

    /// Forward through post-processing.
    public func postProcess(_ x: MLXArray) -> MLXArray {
        var h = post0(x)
        h = post1(h)
        h = post2(h)
        return h
    }
}

// MARK: - Watermark Decoder Block

/// Watermark decoder block with LSTM.
public class DACVAEWatermarkDecoderBlock: Module {
    @ModuleInfo(key: "pre_0") var pre0: DACVAEWNConv1d
    @ModuleInfo(key: "pre_1") var pre1: DACVAELSTMBlock
    @ModuleInfo(key: "post_0") var post0: DACVAEElu
    @ModuleInfo(key: "post_1") var post1: DACVAEWNConv1d

    public init(
        inDim: Int = 128,
        outDim: Int = 1,
        channels: Int = 32,
        hidden: Int = 512,
        lstmLayers: Int = 2
    ) {
        // Pre-processing: Conv + LSTM
        self._pre0.wrappedValue = DACVAEWNConv1d(
            inChannels: inDim,
            outChannels: hidden,
            kernelSize: 7,
            causal: true,
            padMode: "auto",
            norm: "none"
        )
        self._pre1.wrappedValue = DACVAELSTMBlock(
            inputSize: hidden,
            hiddenSize: hidden,
            numLayers: lstmLayers,
            skip: true
        )

        // Post-processing: ELU + Conv
        self._post0.wrappedValue = DACVAEElu()
        self._post1.wrappedValue = DACVAEWNConv1d(
            inChannels: channels,
            outChannels: outDim,
            kernelSize: 7,
            causal: true,
            padMode: "auto",
            norm: "none"
        )
    }

    /// Forward through pre-processing.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = pre0(x)
        h = pre1(h)
        return h
    }

    /// Forward through post-processing.
    public func postProcess(_ x: MLXArray) -> MLXArray {
        var h = post0(x)
        h = post1(h)
        return h
    }
}

// MARK: - Watermarker

/// Watermarking module combining encoder and decoder.
public class DACVAEWatermarker: Module {
    public let nbits: Int

    @ModuleInfo(key: "encoder_block") var encoderBlock: DACVAEWatermarkEncoderBlock
    @ModuleInfo(key: "msg_processor") var msgProcessor: DACVAEMsgProcessor
    @ModuleInfo(key: "decoder_block") var decoderBlock: DACVAEWatermarkDecoderBlock

    public init(
        dOut: Int = 1,
        dLatent: Int = 128,
        channels: Int = 32,
        hidden: Int = 512,
        nbits: Int = 16,
        lstmLayers: Int = 2
    ) {
        self.nbits = nbits

        self._encoderBlock.wrappedValue = DACVAEWatermarkEncoderBlock(
            outDim: dLatent,
            wmChannels: channels,
            hidden: hidden,
            lstmLayers: lstmLayers
        )
        self._msgProcessor.wrappedValue = DACVAEMsgProcessor(nbits: nbits, hiddenSize: dLatent)
        self._decoderBlock.wrappedValue = DACVAEWatermarkDecoderBlock(
            inDim: dLatent,
            outDim: dOut,
            channels: channels,
            hidden: hidden,
            lstmLayers: lstmLayers
        )
    }

    /// Set shared layers from decoder.
    public func setSharedLayers(snakeOut: DACVAESnake1d, convOut: DACVAEWNConv1d) {
        encoderBlock.setSharedLayers(snakeOut: snakeOut, convOut: convOut)
    }

    /// Generate random binary message.
    public func randomMessage(batchSize: Int) -> MLXArray {
        return MLXRandom.randInt(low: 0, high: 2, [batchSize, nbits])
    }
}
