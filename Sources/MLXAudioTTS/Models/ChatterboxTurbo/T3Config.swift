//
//  T3Config.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation

/// Configuration for the T3 (token-to-token) TTS model.
struct T3Config: Sendable {
    // Text tokens
    var startTextToken: Int = 255
    var stopTextToken: Int = 0
    var textTokensDictSize: Int = 50276
    var maxTextTokens: Int = 2048

    // Speech tokens
    var startSpeechToken: Int = 6561
    var stopSpeechToken: Int = 6562
    var speechTokensDictSize: Int = 6563
    var maxSpeechTokens: Int = 4096

    // Model architecture
    var llamaConfigName: String = "GPT2_medium"
    var inputPosEmb: String? = nil
    var speechCondPromptLen: Int = 375

    // Conditioning
    var encoderType: String = "voice_encoder"
    var speakerEmbedSize: Int = 256
    var usePerceiverResampler: Bool = false
    var emotionAdv: Bool = false

    var nChannels: Int {
        GPT2Config.medium.hiddenSize
    }

    var isMultilingual: Bool {
        textTokensDictSize == 2454
    }

    static func turbo() -> T3Config {
        T3Config(
            textTokensDictSize: 50276,
            speechTokensDictSize: 6563,
            llamaConfigName: "GPT2_medium",
            inputPosEmb: nil,
            speechCondPromptLen: 375,
            usePerceiverResampler: false,
            emotionAdv: false
        )
    }
}
