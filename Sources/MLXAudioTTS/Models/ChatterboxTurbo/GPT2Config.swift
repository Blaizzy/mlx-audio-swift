//
//  GPT2Config.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation

/// Configuration for the GPT2 backbone used by T3.
struct GPT2Config: Sendable {
    var vocabSize: Int = 50276
    var nPositions: Int = 8196
    var nEmbeddings: Int = 1024
    var nLayer: Int = 24
    var nHead: Int = 16
    var nInner: Int? = nil
    var activationFunction: String = "gelu_new"
    var residPdrop: Float = 0.1
    var embdPdrop: Float = 0.1
    var attnPdrop: Float = 0.1
    var layerNormEpsilon: Float = 1e-5

    var hiddenSize: Int { nEmbeddings }
    var numAttentionHeads: Int { nHead }
    var numHiddenLayers: Int { nLayer }

    static let medium = GPT2Config(
        vocabSize: 50276,
        nPositions: 8196,
        nEmbeddings: 1024,
        nLayer: 24,
        nHead: 16,
        nInner: nil,
        activationFunction: "gelu_new",
        residPdrop: 0.1,
        embdPdrop: 0.1,
        attnPdrop: 0.1,
        layerNormEpsilon: 1e-5
    )
}
