import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN

public enum DeepFilterNetError: Error, LocalizedError, CustomStringConvertible {
    case invalidRepoID(String)
    case modelPathNotFound(String)
    case missingConfig(URL)
    case missingWeights(URL)
    case missingWeightKey(String)
    case invalidAudioShape([Int])

    public var errorDescription: String? { description }

    public var description: String {
        switch self {
        case .invalidRepoID(let value):
            return "Invalid Hugging Face model repo ID: \(value)"
        case .modelPathNotFound(let path):
            return "Model path not found: \(path)"
        case .missingConfig(let directory):
            return "Missing config.json in model directory: \(directory.path)"
        case .missingWeights(let directory):
            return "Missing .safetensors weights in model directory: \(directory.path)"
        case .missingWeightKey(let key):
            return "Missing DeepFilterNet weight key: \(key)"
        case .invalidAudioShape(let shape):
            return "Expected mono 1D audio array, got shape: \(shape)"
        }
    }
}

public final class DeepFilterNetModel: STSModel {
    public static let defaultRepo = "kylehowells/DeepFilterNet3-MLX"

    public let config: DeepFilterNetConfig
    public let modelDirectory: URL
    public let modelVersion: String
    public var sampleRate: Int { config.sampleRate }

    private let weights: [String: MLXArray]
    private let erbFB: MLXArray
    private let erbInvFB: MLXArray
    private let erbBandWidths: [Int]
    private let vorbisWindow: MLXArray
    private let wnorm: Float
    private let j: MLXArray = MLXArray(real: Float(0.0), imaginary: Float(1.0))

    private init(
        config: DeepFilterNetConfig,
        modelDirectory: URL,
        weights: [String: MLXArray]
    ) throws {
        self.config = config
        self.modelDirectory = modelDirectory
        self.modelVersion = config.modelVersion
        self.weights = weights

        guard let erbFB = weights["erb_fb"], let erbInvFB = weights["mask.erb_inv_fb"] else {
            throw DeepFilterNetError.missingWeightKey("erb_fb / mask.erb_inv_fb")
        }
        self.erbFB = erbFB
        self.erbInvFB = erbInvFB
        let widthsFromConfig = config.erbWidths
        if let widthsFromConfig, widthsFromConfig.reduce(0, +) == config.freqBins {
            self.erbBandWidths = widthsFromConfig
        } else {
            self.erbBandWidths = Self.libdfErbBandWidths(
                sampleRate: config.sampleRate,
                fftSize: config.fftSize,
                nbBands: config.nbErb,
                minNbFreqs: 1
            )
        }
        self.vorbisWindow = Self.vorbisWindow(size: config.fftSize)
        self.wnorm = 1.0 / Float(config.fftSize * config.fftSize) * Float(2 * config.hopSize)
    }

    // MARK: - Loading

    public static func fromPretrained(
        _ modelPathOrRepo: String = defaultRepo,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> DeepFilterNetModel {
        let local = URL(fileURLWithPath: modelPathOrRepo).standardizedFileURL
        if FileManager.default.fileExists(atPath: local.path) {
            if local.hasDirectoryPath {
                return try fromLocal(local)
            }
            return try fromLocal(local.deletingLastPathComponent())
        }

        guard let repoID = Repo.ID(rawValue: modelPathOrRepo) else {
            throw DeepFilterNetError.invalidRepoID(modelPathOrRepo)
        }
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken,
            cache: cache
        )
        return try fromLocal(modelDir)
    }

    public static func fromLocal(_ directory: URL) throws -> DeepFilterNetModel {
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw DeepFilterNetError.missingConfig(directory)
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let configData = try Data(contentsOf: configURL)
        var config = try decoder.decode(DeepFilterNetConfig.self, from: configData)
        if config.modelVersion.isEmpty {
            config.modelVersion = "DeepFilterNet3"
        }

        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard let weightsURL = files.first(where: { $0.lastPathComponent == "model.safetensors" }) ?? files.first else {
            throw DeepFilterNetError.missingWeights(directory)
        }

        let weights = try MLX.loadArrays(url: weightsURL)
        return try DeepFilterNetModel(config: config, modelDirectory: directory, weights: weights)
    }

    // MARK: - Public API

    public func enhance(_ audioInput: MLXArray) throws -> MLXArray {
        guard audioInput.ndim == 1 else {
            throw DeepFilterNetError.invalidAudioShape(audioInput.shape)
        }

        let x = audioInput.asType(.float32)
        let origLen = x.shape[0]
        let padded = MLX.concatenated([
            MLXArray.zeros([config.hopSize], type: Float.self),
            x,
            MLXArray.zeros([config.fftSize], type: Float.self),
        ], axis: 0)

        let specComplex = MossFormer2DSP.stft(
            audio: padded,
            fftLen: config.fftSize,
            hopLength: config.hopSize,
            winLen: config.fftSize,
            window: vorbisWindow,
            center: false
        )
        let spec = specComplex * MLXArray(wnorm)
        let specRe = spec.realPart()
        let specIm = spec.imaginaryPart()

        let specMagSq = specRe.square() + specIm.square()
        let erb = erbEnergies(specMagSq)
        let erbDB = MLXArray(Float(10.0)) * (erb + MLXArray(Float(1e-10))).log10()
        let featErb2D = bandMeanNorm(erbDB)

        let dfRe = specRe[0..., 0..<config.nbDf]
        let dfIm = specIm[0..., 0..<config.nbDf]
        let (dfFeatRe, dfFeatIm) = bandUnitNorm(real: dfRe, imag: dfIm)

        let featErb = featErb2D.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        let featDf = MLX.stacked([dfFeatRe, dfFeatIm], axis: -1)
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)
        let specIn = MLX.stacked([specRe, specIm], axis: -1)
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)

        let (specEnhanced, _, _, _) = try forward(spec: specIn, featErb: featErb, featSpec5D: featDf)
        var enh = specEnhanced[0, 0, 0..., 0..., 0] + j * specEnhanced[0, 0, 0..., 0..., 1]
        enh = enh / MLXArray(wnorm)

        let enhReal = enh.realPart().transposed(1, 0).expandedDimensions(axis: 0)
        let enhImag = enh.imaginaryPart().transposed(1, 0).expandedDimensions(axis: 0)

        var audioOut = MossFormer2DSP.istft(
            real: enhReal,
            imag: enhImag,
            fftLen: config.fftSize,
            hopLength: config.hopSize,
            winLen: config.fftSize,
            window: vorbisWindow,
            center: false,
            audioLength: origLen + config.hopSize + config.fftSize
        )

        let delay = config.fftSize - config.hopSize
        let end = min(delay + origLen, audioOut.shape[0])
        audioOut = audioOut[delay..<end]
        return MLX.clip(audioOut, min: -1.0, max: 1.0)
    }

    // MARK: - Network Forward

    private func forward(
        spec: MLXArray,
        featErb: MLXArray,
        featSpec5D: MLXArray
    ) throws -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let featSpec = featSpec5D
            .squeezed(axis: 1)
            .transposed(0, 3, 1, 2)

        let featErbShift = applyLookahead(feature: featErb, lookahead: config.convLookahead)
        let featSpecShift = applyLookahead(feature: featSpec, lookahead: config.convLookahead)

        let (e0, e1, e2, e3, emb, c0, lsnr) = try encode(featErb: featErbShift, featSpec: featSpecShift)

        let mask = try decodeErb(emb: emb, e3: e3, e2: e2, e1: e1, e0: e0)
        let specMasked = applyMask(spec: spec, mask: mask)

        let dfCoefs = try decodeDf(emb: emb, c0: c0)
        let b = dfCoefs.shape[0]
        let t = dfCoefs.shape[1]
        let dfCoefs5 = dfCoefs
            .reshaped([b, t, config.nbDf, config.dfOrder, 2])
            .transposed(0, 3, 1, 2, 4)

        let specEnhanced: MLXArray
        if config.encConcat {
            specEnhanced = deepFilter(spec: specMasked, coefs: dfCoefs5)
        } else {
            let specDf = deepFilter(spec: spec, coefs: dfCoefs5)
            let low = specDf[0..., 0..., 0..., 0..<config.nbDf, 0...]
            let high = specMasked[0..., 0..., 0..., config.nbDf..., 0...]
            specEnhanced = MLX.concatenated([low, high], axis: 3)
        }

        return (specEnhanced, mask, lsnr, dfCoefs5)
    }

    private func encode(featErb: MLXArray, featSpec: MLXArray)
        throws -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)
    {
        let e0 = try applyEncoderConv(featErb, prefix: "enc.erb_conv0", main: 1, pointwise: nil, bn: 2, fstride: 1)
        let e1 = try applyEncoderConv(e0, prefix: "enc.erb_conv1", main: 0, pointwise: 1, bn: 2, fstride: 2)
        let e2 = try applyEncoderConv(e1, prefix: "enc.erb_conv2", main: 0, pointwise: 1, bn: 2, fstride: 2)
        let e3 = try applyEncoderConv(e2, prefix: "enc.erb_conv3", main: 0, pointwise: 1, bn: 2, fstride: 1)

        let c0 = try applyEncoderConv(featSpec, prefix: "enc.df_conv0", main: 1, pointwise: 2, bn: 3, fstride: 1)
        let c1 = try applyEncoderConv(c0, prefix: "enc.df_conv1", main: 0, pointwise: 1, bn: 2, fstride: 2)

        let b = c1.shape[0]
        let t = c1.shape[2]
        var cemb = c1.transposed(0, 2, 3, 1).reshaped([b, t, -1])
        cemb = relu(groupedLinear(cemb, weight: try w("enc.df_fc_emb.0.weight")))

        var emb = e3.transposed(0, 2, 3, 1).reshaped([b, t, -1])
        emb = config.encConcat ? MLX.concatenated([emb, cemb], axis: -1) : (emb + cemb)

        emb = try squeezedGRU(
            emb,
            prefix: "enc.emb_gru",
            hiddenSize: config.embHiddenDim,
            linearOut: true
        )

        let lsnr = sigmoid(linear(
            emb,
            weight: try w("enc.lsnr_fc.0.weight"),
            bias: try w("enc.lsnr_fc.0.bias")
        )) * MLXArray(Float(config.lsnrMax - config.lsnrMin)) + MLXArray(Float(config.lsnrMin))

        return (e0, e1, e2, e3, emb, c0, lsnr)
    }

    private func decodeErb(
        emb: MLXArray,
        e3: MLXArray,
        e2: MLXArray,
        e1: MLXArray,
        e0: MLXArray
    ) throws -> MLXArray {
        var embDec = try squeezedGRU(
            emb,
            prefix: "erb_dec.emb_gru",
            hiddenSize: config.embHiddenDim,
            linearOut: true
        )

        let b = embDec.shape[0]
        let t = embDec.shape[1]
        let f8 = e3.shape[3]
        embDec = embDec.reshaped([b, t, f8, -1]).transposed(0, 3, 1, 2)

        var d3 = relu(try applyPathwayConv(e3, prefix: "erb_dec.conv3p")) + embDec
        // Matches MLX/PyTorch DF decoder: convt3 is a regular conv block, not transposed.
        d3 = relu(try applyRegularBlock(d3, prefix: "erb_dec.convt3"))
        var d2 = relu(try applyPathwayConv(e2, prefix: "erb_dec.conv2p")) + d3
        d2 = relu(try applyTransposeBlock(d2, prefix: "erb_dec.convt2", fstride: 2))
        var d1 = relu(try applyPathwayConv(e1, prefix: "erb_dec.conv1p")) + d2
        d1 = relu(try applyTransposeBlock(d1, prefix: "erb_dec.convt1", fstride: 2))
        let d0 = relu(try applyPathwayConv(e0, prefix: "erb_dec.conv0p")) + d1
        let out = try applyOutputConv(d0, prefix: "erb_dec.conv0_out")
        return sigmoid(out)
    }

    private func decodeDf(emb: MLXArray, c0: MLXArray) throws -> MLXArray {
        var c = try squeezedGRU(
            emb,
            prefix: "df_dec.df_gru",
            hiddenSize: config.dfHiddenDim,
            linearOut: false
        )

        if weights["df_dec.df_skip.weight"] != nil {
            c = c + groupedLinear(emb, weight: try w("df_dec.df_skip.weight"))
        }

        var c0p = try conv2dLayer(
            c0,
            weight: try w("df_dec.df_convp.1.weight"),
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        c0p = try conv2dLayer(
            c0p,
            weight: try w("df_dec.df_convp.2.weight"),
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        c0p = relu(try batchNorm(c0p, prefix: "df_dec.df_convp.3"))
        c0p = c0p.transposed(0, 2, 3, 1)

        let b = c.shape[0]
        let t = c.shape[1]
        let dfOut = tanh(groupedLinear(c, weight: try w("df_dec.df_out.0.weight")))
            .reshaped([b, t, config.nbDf, config.dfOrder * 2])

        return dfOut + c0p
    }

    private func applyMask(spec: MLXArray, mask: MLXArray) -> MLXArray {
        let b = mask.shape[0]
        let t = mask.shape[2]
        let e = mask.shape[3]
        let flat = mask.reshaped([b * t, e])
        let gains = MLX.matmul(flat, erbInvFB).reshaped([b, 1, t, config.freqBins, 1])
        return spec * gains
    }

    private func deepFilter(spec: MLXArray, coefs: MLXArray) -> MLXArray {
        let t = spec.shape[2]
        let padLeft = config.dfOrder - 1 - config.dfLookahead
        let padRight = config.dfLookahead

        let specLow = spec[0..., 0, 0..., 0..<config.nbDf, 0...]
        let padded = MLX.padded(
            specLow,
            widths: [
                .init(0),
                .init((padLeft, padRight)),
                .init(0),
                .init(0),
            ],
            mode: .constant
        )

        var outFrames = [MLXArray]()
        outFrames.reserveCapacity(t)
        for i in 0..<t {
            let window = padded[0..., i..<(i + config.dfOrder), 0..., 0...]
            let coef = coefs[0..., 0..., i, 0..., 0...]
            let sr = window[0..., 0..., 0..., 0]
            let si = window[0..., 0..., 0..., 1]
            let cr = coef[0..., 0..., 0..., 0]
            let ci = coef[0..., 0..., 0..., 1]

            let outR = MLX.sum(sr * cr - si * ci, axis: 1)
            let outI = MLX.sum(sr * ci + si * cr, axis: 1)
            outFrames.append(MLX.stacked([outR, outI], axis: -1))
        }

        // Stack over time to get [B, 1, T, F_df, 2], matching spec layout.
        let low = MLX.stacked(outFrames, axis: 1).expandedDimensions(axis: 1)
        let high = spec[0..., 0..., 0..., config.nbDf..., 0...]
        return MLX.concatenated([low, high], axis: 3)
    }

    // MARK: - Layer Helpers

    private func applyEncoderConv(
        _ x: MLXArray,
        prefix: String,
        main: Int,
        pointwise: Int?,
        bn: Int,
        fstride: Int
    ) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weight: w("\(prefix).\(main).weight"),
            bias: nil,
            fstride: fstride,
            lookahead: 0
        )
        if let pointwise {
            y = try conv2dLayer(
                y,
                weight: w("\(prefix).\(pointwise).weight"),
                bias: nil,
                fstride: 1,
                lookahead: 0
            )
        }
        y = try batchNorm(y, prefix: "\(prefix).\(bn)")
        return relu(y)
    }

    private func applyPathwayConv(_ x: MLXArray, prefix: String) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weight: w("\(prefix).0.weight"),
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        y = try batchNorm(y, prefix: "\(prefix).1")
        return relu(y)
    }

    private func applyTransposeBlock(_ x: MLXArray, prefix: String, fstride: Int) throws -> MLXArray {
        var y = try convTranspose2dLayer(
            x,
            weight: w("\(prefix).0.weight"),
            fstride: fstride,
            groups: config.convCh
        )
        y = try conv2dLayer(
            y,
            weight: w("\(prefix).1.weight"),
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        return try batchNorm(y, prefix: "\(prefix).2")
    }

    private func applyRegularBlock(_ x: MLXArray, prefix: String) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weight: w("\(prefix).0.weight"),
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        y = try conv2dLayer(
            y,
            weight: w("\(prefix).1.weight"),
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        return try batchNorm(y, prefix: "\(prefix).2")
    }

    private func applyOutputConv(_ x: MLXArray, prefix: String) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weight: w("\(prefix).0.weight"),
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        y = try batchNorm(y, prefix: "\(prefix).1")
        return y
    }

    private func conv2dLayer(
        _ xBCHW: MLXArray,
        weight: MLXArray,
        bias: MLXArray?,
        fstride: Int,
        lookahead: Int
    ) throws -> MLXArray {
        let kT = weight.shape[2]
        let kF = weight.shape[3]
        let inPerGroup = weight.shape[1]
        let inChannels = xBCHW.shape[1]
        let groups = max(1, inChannels / max(1, inPerGroup))

        let rawLeft = kT - 1 - lookahead
        let timeCrop = max(0, -rawLeft)
        let timePadLeft = max(0, rawLeft)
        let timePadRight = max(0, lookahead)
        let freqPad = kF / 2

        var x = xBCHW.transposed(0, 2, 3, 1)  // NHWC
        if timeCrop > 0, x.shape[1] > timeCrop {
            x = x[0..., timeCrop..., 0..., 0...]
        }
        x = MLX.padded(
            x,
            widths: [
                .init(0),
                .init((timePadLeft, timePadRight)),
                .init((freqPad, freqPad)),
                .init(0),
            ],
            mode: .constant
        )

        let wOHWI = weight.transposed(0, 2, 3, 1)
        var y = try groupedConv2d(input: x, weight: wOHWI, strideW: fstride, groups: groups)
        if let bias {
            y = y + bias.reshaped([1, 1, 1, bias.shape[0]])
        }
        return y.transposed(0, 3, 1, 2)
    }

    private func convTranspose2dLayer(
        _ xBCHW: MLXArray,
        weight: MLXArray,
        fstride: Int,
        groups: Int
    ) throws -> MLXArray {
        var x = xBCHW.transposed(0, 2, 3, 1)  // NHWC
        let kT = weight.shape[2]
        let kF = weight.shape[3]
        let padding = IntOrPair((kT - 1, kF / 2))
        let outputPadding = IntOrPair((0, kF / 2))

        if groups <= 1 {
            let w = weight.transposed(1, 2, 3, 0)
            x = MLX.convTransposed2d(
                x,
                w,
                stride: [1, fstride],
                padding: padding,
                outputPadding: outputPadding,
                groups: 1
            )
            return x.transposed(0, 3, 1, 2)
        }

        let inPerGroup = max(1, x.shape[3] / groups)
        let outPerGroup = weight.shape[1]
        var ys = [MLXArray]()
        ys.reserveCapacity(groups)

        for g in 0..<groups {
            let inStart = g * inPerGroup
            let inEnd = inStart + inPerGroup
            let xg = x[0..., 0..., 0..., inStart..<inEnd]

            let wg = weight[inStart..<inEnd, 0..., 0..., 0...]
            let wT = wg.transposed(1, 2, 3, 0)  // [out_pg, kT, kF, in_pg]
            let yg = MLX.convTransposed2d(
                xg,
                wT,
                stride: [1, fstride],
                padding: padding,
                outputPadding: outputPadding,
                groups: 1
            )
            ys.append(yg)
        }

        let y = MLX.concatenated(ys, axis: 3)
        _ = outPerGroup  // keep shape intent explicit
        return y.transposed(0, 3, 1, 2)
    }

    private func groupedConv2d(
        input: MLXArray,
        weight: MLXArray,
        strideW: Int,
        groups: Int
    ) throws -> MLXArray {
        if groups <= 1 {
            return MLX.conv2d(input, weight, stride: [1, strideW], padding: [0, 0], groups: 1)
        }

        let inPerGroup = weight.shape[3]
        let outChannels = weight.shape[0]
        let outPerGroup = max(1, outChannels / groups)
        var ys = [MLXArray]()
        ys.reserveCapacity(groups)
        for g in 0..<groups {
            let inStart = g * inPerGroup
            let inEnd = inStart + inPerGroup
            let outStart = g * outPerGroup
            let outEnd = outStart + outPerGroup
            let xg = input[0..., 0..., 0..., inStart..<inEnd]
            let wg = weight[outStart..<outEnd, 0..., 0..., 0...]
            ys.append(MLX.conv2d(xg, wg, stride: [1, strideW], padding: [0, 0], groups: 1))
        }
        return MLX.concatenated(ys, axis: 3)
    }

    private func batchNorm(_ x: MLXArray, prefix: String) throws -> MLXArray {
        let gamma = try w("\(prefix).weight")
        let beta = try w("\(prefix).bias")
        let mean = try w("\(prefix).running_mean")
        let variance = try w("\(prefix).running_var")

        var y = x.transposed(0, 2, 3, 1)
        y = (y - mean) / MLX.sqrt(variance + MLXArray(Float(1e-5)))
        y = y * gamma + beta
        return y.transposed(0, 3, 1, 2)
    }

    private func squeezedGRU(
        _ x: MLXArray,
        prefix: String,
        hiddenSize: Int,
        linearOut: Bool
    ) throws -> MLXArray {
        var y = relu(groupedLinear(x, weight: try w("\(prefix).linear_in.0.weight")))

        var layer = 0
        while weights["\(prefix).gru.weight_ih_l\(layer)"] != nil {
            y = try pytorchGRULayer(y, prefix: "\(prefix).gru", layer: layer, hiddenSize: hiddenSize)
            layer += 1
        }

        if linearOut, weights["\(prefix).linear_out.0.weight"] != nil {
            y = relu(groupedLinear(y, weight: try w("\(prefix).linear_out.0.weight")))
        }
        return y
    }

    private func groupedLinear(_ x: MLXArray, weight: MLXArray) -> MLXArray {
        let groups = weight.shape[0]
        let ws = weight.shape[1]
        let hs = weight.shape[2]
        let b = x.shape[0]
        let t = x.shape[1]
        let reshaped = x.reshaped([b, t, groups, ws])
        let out = MLX.einsum("btgi,gih->btgh", reshaped, weight)
        return out.reshaped([b, t, groups * hs])
    }

    private func pytorchGRULayer(
        _ x: MLXArray,
        prefix: String,
        layer: Int,
        hiddenSize: Int
    ) throws -> MLXArray {
        let wih = try w("\(prefix).weight_ih_l\(layer)")
        let whh = try w("\(prefix).weight_hh_l\(layer)")
        let bih = try w("\(prefix).bias_ih_l\(layer)")
        let bhh = try w("\(prefix).bias_hh_l\(layer)")

        let b = x.shape[0]
        let t = x.shape[1]

        var h = MLXArray.zeros([b, hiddenSize], type: Float.self)
        var states = [MLXArray]()
        states.reserveCapacity(t)

        for i in 0..<t {
            let xt = x[0..., i, 0...]
            let gx = MLX.addMM(bih, xt, wih.transposed())
            let gh = MLX.addMM(bhh, h, whh.transposed())

            let xr = gx[0..., 0..<hiddenSize]
            let xz = gx[0..., hiddenSize..<(2 * hiddenSize)]
            let xn = gx[0..., (2 * hiddenSize)...]
            let hr = gh[0..., 0..<hiddenSize]
            let hz = gh[0..., hiddenSize..<(2 * hiddenSize)]
            let hn = gh[0..., (2 * hiddenSize)...]

            let r = sigmoid(xr + hr)
            let z = sigmoid(xz + hz)
            let n = tanh(xn + r * hn)
            h = (MLXArray(Float(1.0)) - z) * n + z * h
            states.append(h)
        }
        return MLX.stacked(states, axis: 1)
    }

    private func linear(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        let b = x.shape[0]
        let t = x.shape[1]
        let x2 = x.reshaped([b * t, x.shape[2]])
        var y = MLX.matmul(x2, weight.transposed())
        y = y + bias
        return y.reshaped([b, t, weight.shape[0]])
    }

    // MARK: - Feature Helpers

    private func bandMeanNorm(_ x: MLXArray) -> MLXArray {
        let frames = x.shape[0]
        let bands = x.shape[1]
        let a = normAlpha()
        let oneMinusA = Float(1.0) - a

        let input = x.asArray(Float.self)
        var state = Self.linspace(start: -60.0, end: -90.0, count: bands)
        var out = [Float](repeating: 0, count: input.count)

        for t in 0..<frames {
            let base = t * bands
            for f in 0..<bands {
                let idx = base + f
                state[f] = input[idx] * oneMinusA + state[f] * a
                out[idx] = (input[idx] - state[f]) / 40.0
            }
        }
        return MLXArray(out, [frames, bands])
    }

    private func bandUnitNorm(real: MLXArray, imag: MLXArray) -> (MLXArray, MLXArray) {
        let frames = real.shape[0]
        let bins = real.shape[1]
        let a = normAlpha()
        let oneMinusA = Float(1.0) - a

        let re = real.asArray(Float.self)
        let im = imag.asArray(Float.self)
        var state = Self.linspace(start: 0.001, end: 0.0001, count: bins)
        var outRe = [Float](repeating: 0, count: re.count)
        var outIm = [Float](repeating: 0, count: im.count)

        for t in 0..<frames {
            let base = t * bins
            for f in 0..<bins {
                let idx = base + f
                let mag = sqrt(re[idx] * re[idx] + im[idx] * im[idx])
                state[f] = mag * oneMinusA + state[f] * a
                let denom = sqrt(max(state[f], 1e-12))
                outRe[idx] = re[idx] / denom
                outIm[idx] = im[idx] / denom
            }
        }

        return (MLXArray(outRe, [frames, bins]), MLXArray(outIm, [frames, bins]))
    }

    private func normAlpha() -> Float {
        let aRaw = exp(-Float(config.hopSize) / Float(config.sampleRate))
        var precision = 3
        var a: Float = 1.0
        while a >= 1.0 {
            let scale = powf(10, Float(precision))
            a = (aRaw * scale).rounded() / scale
            precision += 1
        }
        return a
    }

    private func applyLookahead(feature: MLXArray, lookahead: Int) -> MLXArray {
        guard lookahead > 0 else { return feature }
        let t = feature.shape[2]
        guard t > lookahead else { return feature }
        let shifted = feature[0..., 0..., lookahead..<t, 0...]
        let pad = MLXArray.zeros([feature.shape[0], feature.shape[1], lookahead, feature.shape[3]], type: Float.self)
        return MLX.concatenated([shifted, pad], axis: 2)
    }

    // MARK: - Utility

    private func erbEnergies(_ specMagSq: MLXArray) -> MLXArray {
        var bands = [MLXArray]()
        bands.reserveCapacity(erbBandWidths.count)
        var start = 0
        for width in erbBandWidths {
            let stop = min(start + width, config.freqBins)
            if stop > start {
                bands.append(MLX.mean(specMagSq[0..., start..<stop], axis: 1))
            } else {
                bands.append(MLXArray.zeros([specMagSq.shape[0]], type: Float.self))
            }
            start = stop
        }
        return MLX.stacked(bands, axis: 1)
    }

    private static func libdfFreqToErb(_ freqHz: Float) -> Float {
        9.265 * log1p(freqHz / (24.7 * 9.265))
    }

    private static func libdfErbToFreq(_ erb: Float) -> Float {
        24.7 * 9.265 * (exp(erb / 9.265) - 1.0)
    }

    private static func libdfErbBandWidths(
        sampleRate: Int,
        fftSize: Int,
        nbBands: Int,
        minNbFreqs: Int
    ) -> [Int] {
        guard sampleRate > 0, fftSize > 0, nbBands > 0 else { return [] }

        let nyq = sampleRate / 2
        let freqWidth = Float(sampleRate) / Float(fftSize)
        let erbLow = libdfFreqToErb(0)
        let erbHigh = libdfFreqToErb(Float(nyq))
        let step = (erbHigh - erbLow) / Float(nbBands)

        var widths = [Int](repeating: 0, count: nbBands)
        var prevFreq = 0
        var freqOver = 0
        let minBins = max(1, minNbFreqs)

        for i in 1...nbBands {
            let f = libdfErbToFreq(erbLow + Float(i) * step)
            let fb = Int((f / freqWidth).rounded())
            var nbFreqs = fb - prevFreq - freqOver
            if nbFreqs < minBins {
                freqOver = minBins - nbFreqs
                nbFreqs = minBins
            } else {
                freqOver = 0
            }
            widths[i - 1] = max(1, nbFreqs)
            prevFreq = fb
        }

        widths[nbBands - 1] += 1  // fft_size/2 + 1 bins
        let target = fftSize / 2 + 1
        let total = widths.reduce(0, +)
        if total > target {
            widths[nbBands - 1] -= (total - target)
        } else if total < target {
            widths[nbBands - 1] += (target - total)
        }
        return widths
    }

    private static func vorbisWindow(size: Int) -> MLXArray {
        let half = max(1, size / 2)
        var window = [Float](repeating: 0, count: size)
        for i in 0..<size {
            let inner = sin(0.5 * Float.pi * (Float(i) + 0.5) / Float(half))
            window[i] = sin(0.5 * Float.pi * inner * inner)
        }
        return MLXArray(window)
    }

    private static func linspace(start: Float, end: Float, count: Int) -> [Float] {
        guard count > 1 else { return [start] }
        let step = (end - start) / Float(count - 1)
        return (0..<count).map { start + Float($0) * step }
    }

    private func w(_ key: String) throws -> MLXArray {
        guard let value = weights[key] else {
            throw DeepFilterNetError.missingWeightKey(key)
        }
        return value
    }
}
