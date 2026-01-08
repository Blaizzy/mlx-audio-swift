//
//  T3Cond.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation
@preconcurrency import MLX
import MLXNN

/// Conditioning container for T3.
struct T3Cond {
    var speakerEmb: MLXArray
    var clapEmb: MLXArray?
    var condPromptSpeechTokens: MLXArray?
    var condPromptSpeechEmb: MLXArray?
    var emotionAdv: MLXArray?
}

/// Conditioning encoder for T3.
final class T3CondEnc: Module {
    private let hp: T3Config

    @ModuleInfo(key: "spkr_enc") private var spkrEnc: Linear
    @ModuleInfo(key: "emotion_adv_fc") private var emotionAdvFc: Linear?

    init(_ hp: T3Config) {
        self.hp = hp

        self._spkrEnc.wrappedValue = Linear(hp.speakerEmbedSize, hp.nChannels)

        if hp.emotionAdv {
            self._emotionAdvFc.wrappedValue = Linear(1, hp.nChannels, bias: false)
        } else {
            self._emotionAdvFc.wrappedValue = nil
        }
    }

    func callAsFunction(_ cond: T3Cond) -> MLXArray {
        let speakerEmb = cond.speakerEmb.reshaped(-1, hp.speakerEmbedSize)
        let condSpkr = spkrEnc(speakerEmb).expandedDimensions(axis: 1)

        let batch = condSpkr.dim(0)
        let dim = condSpkr.dim(2)

        let empty = MLXArray.zeros([batch, 0, dim], type: Float.self)

        precondition(cond.clapEmb == nil, "clap_embed not implemented")
        let condClap = empty

        var condPromptSpeechEmb = cond.condPromptSpeechEmb ?? empty

        if hp.usePerceiverResampler {
            fatalError("Perceiver resampler not implemented")
        }

        var condEmotionAdv = empty
        if hp.emotionAdv,
           let emotionAdv = cond.emotionAdv,
           let emotionAdvFc = emotionAdvFc
        {
            let emotionVal = emotionAdv.reshaped(-1, 1, 1)
            condEmotionAdv = emotionAdvFc(emotionVal)
        }

        return MLX.concatenated([condSpkr, condClap, condPromptSpeechEmb, condEmotionAdv], axis: 1)
    }
}
