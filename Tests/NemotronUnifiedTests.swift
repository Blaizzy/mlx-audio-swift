import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudioCore
@testable import MLXAudioSTT

/// Nemotron Unified (buffered streaming) on tiny random models, small enough to run
/// on any Metal device, so CI runs them without the MLX runtime-test opt-in.
struct NemotronUnifiedTests {
    private func unifiedConfigJSON() -> String {
        """
        {
          "model_type": "nemotron_asr",
          "preprocessor": {
            "sample_rate": 16000,
            "features": 16,
            "n_fft": 64,
            "window_size": 0.004,
            "window_stride": 0.002,
            "window": "hann",
            "preemph": 0.97,
            "dither": 0.0,
            "normalize": "per_feature"
          },
          "encoder": {
            "feat_in": 16,
            "n_layers": 1,
            "d_model": 16,
            "n_heads": 2,
            "ff_expansion_factor": 2,
            "subsampling_factor": 4,
            "subsampling_conv_channels": 4,
            "conv_kernel_size": 3,
            "causal_downsampling": false,
            "conv_context_size": [1, 1],
            "conv_norm_type": "batch_norm",
            "self_attention_model": "rel_pos",
            "att_context_style": "chunked_limited_with_rc",
            "att_context_size": [[-1, -1, -1]],
            "pos_emb_max_len": 64,
            "use_bias": true,
            "xscaling": true
          },
          "prompt": {
            "num_prompts": 4,
            "prompt_hidden": 16,
            "prompt_dictionary": {"en-US": 0, "auto": 1}
          },
          "decoder": {
            "pred_hidden": 8,
            "pred_rnn_layers": 1,
            "vocab_size": 6,
            "blank_as_pad": true
          },
          "joint": {
            "joint_hidden": 8,
            "activation": "relu",
            "encoder_hidden": 16,
            "pred_hidden": 8,
            "num_classes": 6
          },
          "vocabulary": ["<unk>", "<en-US>", "▁hello", "▁world", "!", "a"],
          "default_language": "auto",
          "default_att_context_size": [4, 1, 2],
          "max_symbols": 3
        }
        """
    }

    private func unifiedModel() throws -> NemotronASRModel {
        let config = try JSONDecoder().decode(NemotronASRConfig.self, from: Data(unifiedConfigJSON().utf8))
        let model = NemotronASRModel(config)
        eval(model.parameters())
        model.train(false)
        return model
    }

    @Test func unifiedConfigSelectsBufferedStreaming() throws {
        let config = try JSONDecoder().decode(NemotronASRConfig.self, from: Data(unifiedConfigJSON().utf8))
        #expect(config.encoder.attContextStyle == "chunked_limited_with_rc")
        #expect(config.encoder.convContextSize == .explicit(left: 1, right: 1))
        #expect(config.defaultAttContextSize == [4, 1, 2])
        let cases: [(Int, Int, Int)] = [
            (80, 1, 1), (160, 1, 1), (200, 1, 2), (320, 1, 3), (400, 1, 4), (480, 1, 5),
            (560, 2, 5), (1040, 2, 11), (1120, 7, 7), (1200, 7, 8), (2080, 13, 13), (5000, 13, 13),
        ]
        for (latency, chunk, right) in cases {
            #expect(NemotronASRBufferedContext.forLatency(latency).asArray == [70, chunk, right])
        }
        #expect(NemotronASRBufferedContext.forLatency(1120, left: 25).asArray == [25, 7, 7])
    }

    /// Layouts come from callers, so a bad one is an error, not a trap.
    @Test func aBufferedContextRejectsAnInvalidLayout() throws {
        #expect(throws: NemotronASRStreamingError.invalidBufferedContext(left: 70, chunk: 0, right: 2)) {
            try NemotronASRBufferedContext(left: 70, chunk: 0, right: 2)
        }
        #expect(throws: NemotronASRStreamingError.invalidBufferedContext(left: -1, chunk: 1, right: 0)) {
            try NemotronASRBufferedContext(left: -1, chunk: 1, right: 0)
        }
        #expect(try NemotronASRBufferedContext(left: 0, chunk: 1, right: 0).asArray == [0, 1, 0])
    }

    /// A Unified config without `default_att_context_size` streams at 320 ms, not at
    /// its full-context training layout.
    @Test func aUnifiedConfigWithoutADefaultLayoutStreamsAt320Ms() throws {
        let json = unifiedConfigJSON()
            .replacingOccurrences(of: "\"default_att_context_size\": [4, 1, 2],", with: "")
        let config = try JSONDecoder().decode(NemotronASRConfig.self, from: Data(json.utf8))
        #expect(config.defaultAttContextSize == [70, 1, 3])
    }

    /// An explicit layout on a cache-aware (3.5-style) checkpoint is an error.
    @Test func anExplicitLayoutOnACacheAwareCheckpointThrows() throws {
        let json = unifiedConfigJSON()
            .replacingOccurrences(of: "\"att_context_style\": \"chunked_limited_with_rc\"", with: "\"att_context_style\": \"chunked_limited\"")
            .replacingOccurrences(of: "\"att_context_size\": [[-1, -1, -1]]", with: "\"att_context_size\": [[4, 1]]")
            .replacingOccurrences(of: "\"default_att_context_size\": [4, 1, 2]", with: "\"default_att_context_size\": [4, 1]")
        let model = NemotronASRModel(try JSONDecoder().decode(NemotronASRConfig.self, from: Data(json.utf8)))
        #expect(!model.isBuffered)
        #expect(throws: NemotronASRStreamingError.notBuffered) {
            try model.makeStreamSession(language: nil, bufferedContext: .forLatency(320))
        }
    }

    @Test func chunkedRCVisibilityMatchesNeMo() {
        // left 2, chunk 2, right 1: frame i sees [chunkStart - 2, chunkStart + 2].
        let visible = NemotronASRAttentionMask.chunkedRCVisibility(seqLen: 7, left: 2, chunk: 2, right: 1)
        let rows = visible.asType(.int32).asArray(Int32.self)
        func row(_ i: Int) -> [Int32] { Array(rows[(i * 7)..<(i * 7 + 7)]) }
        #expect(row(0) == [1, 1, 1, 0, 0, 0, 0])
        #expect(row(3) == [1, 1, 1, 1, 1, 0, 0])
        #expect(row(6) == [0, 0, 0, 0, 1, 1, 1])
    }

    /// NeMo masks padding in the subsampling, the attention keys and the conv, so
    /// frames past the valid length cannot change the valid frames' output.
    @Test func unifiedEncoderIgnoresPaddedFrames() throws {
        let model = try unifiedModel()
        let mel = MLXRandom.normal([1, 40, 16], key: MLXRandom.key(3))
        let noise = MLXRandom.normal([1, 12, 16], key: MLXRandom.key(4))
        let padded = MLX.concatenated([mel, noise], axis: 1)
        let lengths = MLXArray([Int32(40)])
        for context in [[-1, -1, -1], [4, 1, 2]] {
            let (a, la) = model.encoder(mel, lengths: lengths, attContextSize: context)
            let (b, lb) = model.encoder(padded, lengths: lengths, attContextSize: context)
            let valid = Int(la[0].item(Int32.self))
            #expect(valid == Int(lb[0].item(Int32.self)))
            let diff = MLX.abs(a[0..., ..<valid, 0...] - b[0..., ..<valid, 0...]).max().item(Float.self)
            #expect(diff < 1e-4)
        }
    }

    /// A left-0 layout leaves a padded chunk with every key masked. A mask value that
    /// overflows to -inf in float16 turns such rows NaN, and later layers spread the
    /// NaN to valid frames (0 x NaN).
    @Test func float16MaskStaysFiniteWhenAChunkIsAllPadding() throws {
        let model = try unifiedModel()
        let half = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { key, value in
                (key, value.dtype.isFloatingPoint ? value.asType(.float16) : value)
            }
        )
        try model.update(parameters: ModuleParameters.unflattened(half), verify: .noUnusedKeys)
        let mel = MLXRandom.normal([1, 40, 16], key: MLXRandom.key(3))
        let noise = MLXRandom.normal([1, 12, 16], key: MLXRandom.key(4))
        let padded = MLX.concatenated([mel, noise], axis: 1).asType(.float16)
        let (out, lengths) = model.encoder(padded, lengths: MLXArray([Int32(40)]), attContextSize: [0, 2, 0])
        let valid = Int(lengths[0].item(Int32.self))
        #expect(valid > 0)
        // x - x is 0 for finite values and NaN for NaN or infinity.
        let frames = out[0..., ..<valid, 0...].asType(.float32)
        #expect(MLX.abs(frames - frames).sum().item(Float.self) == 0)
    }

    @Test func perFeatureMelNormalizesValidFramesAndZeroesTheTail() throws {
        let config = try JSONDecoder().decode(NemotronASRConfig.self, from: Data(unifiedConfigJSON().utf8)).preprocessor
        let samples = 40 * config.hopLength + 7
        let mel = NemotronASRAudio.logMelSpectrogram(syntheticAudio(samples: samples), config: config)[0]
        #expect(mel.shape[0] == 41)  // centered STFT: samples / hop + 1 frames
        #expect(MLX.abs(mel[40]).max().item(Float.self) == 0)
        let mean = MLX.abs(MLX.mean(mel[..<40], axis: 0)).max().item(Float.self)
        #expect(mean < 1e-4)
    }

    /// A stop with under one hop of new audio leaves no frame to decode.
    @Test func unifiedSessionStopsCleanlyOnTinyInput() async throws {
        let model = try unifiedModel()
        let frame = model.encoderConfig.subsamplingFactor * model.preprocessConfig.hopLength
        for length in [1, 16, 31, 32, 33, 7 * frame + 5] {
            let audio = syntheticAudio(samples: length)
            let whole = try await wholeStreamText(model, audio)
            let (text, _) = unifiedSessionText(model, audio, feed: 61)
            #expect(text == whole)
        }
    }

    @Test func unifiedSessionTakesAnExplicitContext() throws {
        let model = try unifiedModel()
        let samples = syntheticAudio(samples: 20 * 128 + 70).asArray(Float.self)
        let context = try NemotronASRBufferedContext(left: 2, chunk: 2, right: 1)
        func tokens(feed: Int) throws -> [Int] {
            let session = try model.makeStreamSession(language: "en-US", bufferedContext: context)
            var i = 0
            while i < samples.count {
                let e = min(i + feed, samples.count)
                _ = session.step(Array(samples[i..<e]))
                i = e
            }
            _ = session.finish()
            return session.tokens
        }
        #expect(try tokens(feed: 61) == tokens(feed: 1000))
    }

    @Test func promptLanguageResolvesLikeNeMo() throws {
        let model = try unifiedModel()  // prompts: en-US 0, auto 1; default auto
        #expect(model.resolvePromptIndex(nil) == 1)
        #expect(model.resolvePromptIndex("en-US") == 0)
        #expect(model.resolvePromptIndex("en") == 0)
        #expect(model.resolvePromptIndex("EN") == 0)
        #expect(model.resolvePromptIndex("fr") == 1)  // unsupported: the default language
    }

    /// Every prompt_dictionary key of nvidia/nemotron-labs-unified-asr-0.6b@023ce907, its
    /// bare code and case variants resolve to the index NeMo's `resolve_prompt_index`
    /// returns (null: NeMo raises; the port then uses the default language, `auto`).
    @Test func promptLanguageTableMatchesNeMo() throws {
        let dictionary = #"{"en-US": 0, "en": 0, "English": 0, "en-GB": 1, "es-ES": 2, "es": 3, "zh-CN": 4, "zh-ZH": 4, "zh-TW": 5, "hi-IN": 6, "hi": 6, "hi-HI": 6, "ar-AR": 7, "ar": 7, "fr-FR": 8, "fr": 8, "de-DE": 9, "de": 9, "ja-JP": 10, "ja-JA": 10, "ru-RU": 11, "ru": 11, "pt-BR": 12, "pt-PT": 13, "pt": 13, "ko-KR": 14, "ko": 14, "ko-KO": 14, "it-IT": 15, "it": 15, "nl-NL": 16, "nl": 16, "pl-PL": 17, "pl": 17, "tr-TR": 18, "tr": 18, "uk-UA": 19, "uk": 19, "ro-RO": 20, "ro": 20, "el-GR": 21, "el": 21, "cs-CZ": 22, "cs": 22, "hu-HU": 23, "hu": 23, "sv-SE": 24, "sv": 24, "da-DK": 25, "da": 25, "fi-FI": 26, "fi": 26, "no-NO": 27, "no": 27, "nb-NO": 103, "nb": 103, "sk-SK": 28, "sk": 28, "hr-HR": 29, "hr": 29, "bg-BG": 30, "bg": 30, "lt-LT": 31, "lt": 31, "et-EE": 60, "et": 60, "lv-LV": 61, "lv": 61, "sl-SI": 62, "sl": 62, "th-TH": 32, "vi-VN": 33, "id-ID": 34, "ms-MY": 35, "bn-IN": 36, "ur-PK": 37, "fa-IR": 38, "ta-IN": 39, "te-IN": 40, "mr-IN": 41, "gu-IN": 42, "kn-IN": 43, "ml-IN": 44, "si-LK": 45, "ne-NP": 46, "km-KH": 47, "sw-KE": 48, "am-ET": 49, "ha-NG": 50, "zu-ZA": 51, "yo-NG": 52, "ig-NG": 53, "af-ZA": 54, "rw-RW": 55, "so-SO": 56, "ny-MW": 57, "ln-CD": 58, "or-KE": 59, "he-IL": 64, "ku-TR": 65, "az-AZ": 66, "ka-GE": 67, "hy-AM": 68, "uz-UZ": 69, "tg-TJ": 70, "ky-KG": 71, "qu-PE": 80, "ay-BO": 81, "gn-PY": 82, "nah-MX": 83, "mi-NZ": 96, "haw-US": 97, "sm-WS": 98, "to-TO": 99, "fr-CA": 100, "mt-MT": 102, "mt": 102, "auto": 127}"#
        let nemo = #"{"AF":54,"AF-ZA":null,"AM":49,"AM-ET":null,"AR":7,"AR-AR":null,"AUTO":null,"AY":81,"AY-BO":null,"AZ":66,"AZ-AZ":null,"Af":54,"Am":49,"Ar":7,"Auto":null,"Ay":81,"Az":66,"BG":30,"BG-BG":null,"BN":36,"BN-IN":null,"Bg":30,"Bn":36,"CS":22,"CS-CZ":null,"Cs":22,"DA":25,"DA-DK":null,"DE":9,"DE-DE":null,"Da":25,"De":9,"EL":21,"EL-GR":null,"EN":null,"EN-GB":null,"EN-US":null,"ENGLISH":null,"ES":2,"ES-ES":null,"ET":60,"ET-EE":null,"El":21,"En":null,"English":0,"Es":2,"Et":60,"FA":38,"FA-IR":null,"FI":26,"FI-FI":null,"FR":null,"FR-CA":null,"FR-FR":null,"Fa":38,"Fi":26,"Fr":null,"GN":82,"GN-PY":null,"GU":42,"GU-IN":null,"Gn":82,"Gu":42,"HA":50,"HA-NG":null,"HAW":97,"HAW-US":null,"HE":64,"HE-IL":null,"HI":null,"HI-HI":null,"HI-IN":null,"HR":29,"HR-HR":null,"HU":23,"HU-HU":null,"HY":68,"HY-AM":null,"Ha":50,"Haw":97,"He":64,"Hi":null,"Hr":29,"Hu":23,"Hy":68,"ID":34,"ID-ID":null,"IG":53,"IG-NG":null,"IT":15,"IT-IT":null,"Id":34,"Ig":53,"It":15,"JA":null,"JA-JA":null,"JA-JP":null,"Ja":null,"KA":67,"KA-GE":null,"KM":47,"KM-KH":null,"KN":43,"KN-IN":null,"KO":null,"KO-KO":null,"KO-KR":null,"KU":65,"KU-TR":null,"KY":71,"KY-KG":null,"Ka":67,"Km":47,"Kn":43,"Ko":null,"Ku":65,"Ky":71,"LN":58,"LN-CD":null,"LT":31,"LT-LT":null,"LV":61,"LV-LV":null,"Ln":58,"Lt":31,"Lv":61,"MI":96,"MI-NZ":null,"ML":44,"ML-IN":null,"MR":41,"MR-IN":null,"MS":35,"MS-MY":null,"MT":102,"MT-MT":null,"Mi":96,"Ml":44,"Mr":41,"Ms":35,"Mt":102,"NAH":83,"NAH-MX":null,"NB":103,"NB-NO":null,"NE":46,"NE-NP":null,"NL":16,"NL-NL":null,"NO":27,"NO-NO":null,"NY":57,"NY-MW":null,"Nah":83,"Nb":103,"Ne":46,"Nl":16,"No":27,"Ny":57,"OR":59,"OR-KE":null,"Or":59,"PL":17,"PL-PL":null,"PT":null,"PT-BR":null,"PT-PT":null,"Pl":17,"Pt":null,"QU":80,"QU-PE":null,"Qu":80,"RO":20,"RO-RO":null,"RU":11,"RU-RU":null,"RW":55,"RW-RW":null,"Ro":20,"Ru":11,"Rw":55,"SI":45,"SI-LK":null,"SK":28,"SK-SK":null,"SL":62,"SL-SI":null,"SM":98,"SM-WS":null,"SO":56,"SO-SO":null,"SV":24,"SV-SE":null,"SW":48,"SW-KE":null,"Si":45,"Sk":28,"Sl":62,"Sm":98,"So":56,"Sv":24,"Sw":48,"TA":39,"TA-IN":null,"TE":40,"TE-IN":null,"TG":70,"TG-TJ":null,"TH":32,"TH-TH":null,"TO":99,"TO-TO":null,"TR":18,"TR-TR":null,"Ta":39,"Te":40,"Tg":70,"Th":32,"To":99,"Tr":18,"UK":19,"UK-UA":null,"UR":37,"UR-PK":null,"UZ":69,"UZ-UZ":null,"Uk":19,"Ur":37,"Uz":69,"VI":33,"VI-VN":null,"Vi":33,"YO":52,"YO-NG":null,"Yo":52,"ZH":null,"ZH-CN":null,"ZH-TW":null,"ZH-ZH":null,"ZU":51,"ZU-ZA":null,"Zh":null,"Zu":51,"af":54,"af-ZA":54,"af-za":null,"am":49,"am-ET":49,"am-et":null,"ar":7,"ar-AR":7,"ar-ar":null,"auto":127,"ay":81,"ay-BO":81,"ay-bo":null,"az":66,"az-AZ":66,"az-az":null,"bg":30,"bg-BG":30,"bg-bg":null,"bn":36,"bn-IN":36,"bn-in":null,"cs":22,"cs-CZ":22,"cs-cz":null,"da":25,"da-DK":25,"da-dk":null,"de":9,"de-DE":9,"de-de":null,"el":21,"el-GR":21,"el-gr":null,"en":0,"en-GB":1,"en-US":0,"en-gb":null,"en-us":null,"english":null,"es":3,"es-ES":2,"es-MX":null,"es-es":null,"et":60,"et-EE":60,"et-ee":null,"fa":38,"fa-IR":38,"fa-ir":null,"fi":26,"fi-FI":26,"fi-fi":null,"fr":8,"fr-CA":100,"fr-FR":8,"fr-ca":null,"fr-fr":null,"gn":82,"gn-PY":82,"gn-py":null,"gu":42,"gu-IN":42,"gu-in":null,"ha":50,"ha-NG":50,"ha-ng":null,"haw":97,"haw-US":97,"haw-us":null,"he":64,"he-IL":64,"he-il":null,"hi":6,"hi-HI":6,"hi-IN":6,"hi-hi":null,"hi-in":null,"hr":29,"hr-HR":29,"hr-hr":null,"hu":23,"hu-HU":23,"hu-hu":null,"hy":68,"hy-AM":68,"hy-am":null,"id":34,"id-ID":34,"id-id":null,"ig":53,"ig-NG":53,"ig-ng":null,"it":15,"it-IT":15,"it-it":null,"ja":null,"ja-JA":10,"ja-JP":10,"ja-ja":null,"ja-jp":null,"ka":67,"ka-GE":67,"ka-ge":null,"km":47,"km-KH":47,"km-kh":null,"kn":43,"kn-IN":43,"kn-in":null,"ko":14,"ko-KO":14,"ko-KR":14,"ko-ko":null,"ko-kr":null,"ku":65,"ku-TR":65,"ku-tr":null,"ky":71,"ky-KG":71,"ky-kg":null,"ln":58,"ln-CD":58,"ln-cd":null,"lt":31,"lt-LT":31,"lt-lt":null,"lv":61,"lv-LV":61,"lv-lv":null,"mi":96,"mi-NZ":96,"mi-nz":null,"ml":44,"ml-IN":44,"ml-in":null,"mr":41,"mr-IN":41,"mr-in":null,"ms":35,"ms-MY":35,"ms-my":null,"mt":102,"mt-MT":102,"mt-mt":null,"nah":83,"nah-MX":83,"nah-mx":null,"nb":103,"nb-NO":103,"nb-no":null,"ne":46,"ne-NP":46,"ne-np":null,"nl":16,"nl-NL":16,"nl-nl":null,"no":27,"no-NO":27,"no-no":null,"ny":57,"ny-MW":57,"ny-mw":null,"or":59,"or-KE":59,"or-ke":null,"pl":17,"pl-PL":17,"pl-pl":null,"pt":13,"pt-BR":12,"pt-PT":13,"pt-br":null,"pt-pt":null,"qu":80,"qu-PE":80,"qu-pe":null,"ro":20,"ro-RO":20,"ro-ro":null,"ru":11,"ru-RU":11,"ru-ru":null,"rw":55,"rw-RW":55,"rw-rw":null,"si":45,"si-LK":45,"si-lk":null,"sk":28,"sk-SK":28,"sk-sk":null,"sl":62,"sl-SI":62,"sl-si":null,"sm":98,"sm-WS":98,"sm-ws":null,"so":56,"so-SO":56,"so-so":null,"sv":24,"sv-SE":24,"sv-se":null,"sw":48,"sw-KE":48,"sw-ke":null,"ta":39,"ta-IN":39,"ta-in":null,"te":40,"te-IN":40,"te-in":null,"tg":70,"tg-TJ":70,"tg-tj":null,"th":32,"th-TH":32,"th-th":null,"to":99,"to-TO":99,"to-to":null,"tr":18,"tr-TR":18,"tr-tr":null,"uk":19,"uk-UA":19,"uk-ua":null,"ur":37,"ur-PK":37,"ur-pk":null,"uz":69,"uz-UZ":69,"uz-uz":null,"vi":33,"vi-VN":33,"vi-vn":null,"xx":null,"yo":52,"yo-NG":52,"yo-ng":null,"zh":null,"zh-CN":4,"zh-TW":5,"zh-ZH":4,"zh-cn":null,"zh-tw":null,"zh-zh":null,"zu":51,"zu-ZA":51,"zu-za":null}"#
        let json = unifiedConfigJSON()
            .replacingOccurrences(of: "\"num_prompts\": 4", with: "\"num_prompts\": 128")
            .replacingOccurrences(of: "\"prompt_dictionary\": {\"en-US\": 0, \"auto\": 1}", with: "\"prompt_dictionary\": " + dictionary)
        let model = NemotronASRModel(try JSONDecoder().decode(NemotronASRConfig.self, from: Data(json.utf8)))
        #expect(model.promptDictionary.count == 118)
        let expected = try JSONSerialization.jsonObject(with: Data(nemo.utf8)) as! [String: Any]
        for (language, index) in expected {
            #expect(model.resolvePromptIndex(language) == ((index as? Int) ?? 127), "\(language)")
        }
        #expect(model.resolvePromptIndex(nil) == 127)
        #expect(["es", "es-ES", "pt", "pt-BR", "en", "en-GB"].map { model.resolvePromptIndex($0) } == [3, 2, 13, 12, 0, 1])
    }

    private func unifiedSessionText(_ model: NemotronASRModel, _ audio: MLXArray, feed: Int) -> (String, [Int]) {
        let samples = audio.asArray(Float.self)
        let session = model.makeStreamSession(language: "en-US")
        var i = 0
        while i < samples.count {
            let e = min(i + feed, samples.count)
            _ = session.step(Array(samples[i..<e]))
            i = e
        }
        _ = session.finish()
        return (session.text, session.tokens)
    }

    /// The buffered session segments a live stream as NeMo segments the whole file,
    /// whatever the feed size, so it matches `generateStream` token for token.
    @Test func unifiedSessionMatchesGenerateStreamAtAnyFeedSize() async throws {
        let model = try unifiedModel()
        let frame = model.encoderConfig.subsamplingFactor * model.preprocessConfig.hopLength
        // Lengths ending mid-frame, on a chunk boundary, and on chunk + right context.
        for length in [9 * frame + 37, 12 * frame, 15 * frame, 2 * frame + 1] {
            let audio = syntheticAudio(samples: length)
            let whole = try await wholeStreamText(model, audio)
            let (fine, fineTokens) = unifiedSessionText(model, audio, feed: 61)
            let (coarse, coarseTokens) = unifiedSessionText(model, audio, feed: 5 * frame)
            #expect(fine == whole)
            #expect(fineTokens == coarseTokens)
            #expect(coarse == whole)
        }
    }


    /// Steps `samples` through `session`, `feed` samples at a time. After every step the
    /// text only grows, by exactly the returned delta, as a realtime server relies on.
    private func appendOnlyTokens(
        _ session: NemotronASRStreamSession, _ samples: [Float], feed: Int
    ) -> [Int] {
        var joined = ""
        var i = 0
        while i < samples.count {
            let e = min(i + feed, samples.count)
            let before = session.text
            joined += session.step(Array(samples[i..<e])).text
            #expect(session.text.hasPrefix(before))
            #expect(joined == session.text)
            i = e
        }
        joined += session.finish().text
        #expect(joined == session.text)
        return session.tokens
    }

    /// Chunks of 1, 2 and 7 frames, past the startup ramp into full windows.
    @Test func unifiedSessionTextOnlyGrowsAtEveryChunkSize() throws {
        let model = try unifiedModel()
        let frame = model.encoderConfig.subsamplingFactor * model.preprocessConfig.hopLength
        let samples = syntheticAudio(samples: 40 * frame + 77).asArray(Float.self)
        for latency in [160, 320, 560, 1120] {
            let runs = [61, frame, 23 * frame].map { feed in
                appendOnlyTokens(
                    model.makeStreamSession(language: "en-US", chunkMs: latency), samples, feed: feed
                )
            }
            #expect(runs[1] == runs[0], "latency \(latency) ms")
            #expect(runs[2] == runs[0], "latency \(latency) ms")
        }
    }

    /// Encoders with different layouts and languages, pushed alternately on one model,
    /// must each give what they give alone.
    @Test func interleavedUnifiedEncodersDoNotShareState() throws {
        let model = try unifiedModel()
        let frame = model.encoderConfig.subsamplingFactor * model.preprocessConfig.hopLength
        let a = syntheticAudio(samples: 40 * frame + 50).asArray(Float.self)
        let streams: [(context: NemotronASRBufferedContext, language: String, samples: [Float])] = [
            (try NemotronASRBufferedContext(left: 4, chunk: 2, right: 2), "en-US", a),
            (try NemotronASRBufferedContext(left: 4, chunk: 7, right: 7), "auto", Array(a.reversed())),
        ]
        func encoder(_ s: Int) -> NemotronASRBufferedEncoder {
            NemotronASRBufferedEncoder(model: model, context: streams[s].context, language: streams[s].language)
        }
        func push(_ e: NemotronASRBufferedEncoder, _ s: Int, at i: Int) -> [MLXArray] {
            let end = min(i + frame, streams[s].samples.count)
            return e.push(Array(streams[s].samples[i..<end]), final: end == streams[s].samples.count)
        }
        let alone = (0..<2).map { s -> [MLXArray] in
            let e = encoder(s)
            return stride(from: 0, to: a.count, by: frame).flatMap { push(e, s, at: $0) }
        }
        let encoders = [encoder(0), encoder(1)]
        var together: [[MLXArray]] = [[], []]
        for i in stride(from: 0, to: a.count, by: frame) {
            for s in 0..<2 { together[s] += push(encoders[s], s, at: i) }
        }
        for s in 0..<2 {
            #expect(!alone[s].isEmpty)
            #expect(together[s].count == alone[s].count)
            for (x, y) in zip(together[s], alone[s]) {
                #expect(MLX.abs(x - y).max().item(Float.self) < 1e-5)
            }
        }
    }

    private func syntheticAudio(samples: Int) -> MLXArray {
        let values = (0..<samples).map { i in
            0.1 * sin(Float(i) * 0.05) + 0.05 * sin(Float(i) * 0.17)
        }
        return MLXArray(values)
    }

    private func wholeStreamText(_ model: NemotronASRModel, _ audio: MLXArray) async throws -> String {
        var text = ""
        for try await event in model.generateStream(
            audio: audio,
            generationParameters: STTGenerateParameters(language: "en-US")
        ) {
            if case let .result(out) = event { text = out.text }
        }
        return text
    }
}
