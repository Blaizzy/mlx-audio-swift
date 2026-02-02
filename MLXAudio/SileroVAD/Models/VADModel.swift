import Foundation
import MLX
import MLXNN
import MLXRandom

/// Custom Conv1d layer with mutable weight and bias for weight loading
final class VADConv1d: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?
    let padding: Int
    let stride: Int

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int, stride: Int = 1, padding: Int = 0, bias: Bool = true) {
        let scale = sqrt(1 / Float(inputChannels * kernelSize))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [outputChannels, kernelSize, inputChannels])
        self._bias.wrappedValue = bias ? MLXArray.zeros([outputChannels]) : nil
        self.padding = padding
        self.stride = stride
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1d(x, weight, stride: stride, padding: padding)
        if let bias {
            y = y + bias
        }
        return y
    }

    func setWeights(weight: MLXArray, bias: MLXArray? = nil) {
        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }
}

/// Custom LSTM layer with mutable weights for weight loading
final class VADLSTM: Module {
    @ParameterInfo(key: "Wx") var wx: MLXArray
    @ParameterInfo(key: "Wh") var wh: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?

    init(inputSize: Int, hiddenSize: Int, bias: Bool = true) {
        let scale = 1 / sqrt(Float(hiddenSize))
        self._wx.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [4 * hiddenSize, inputSize])
        self._wh.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [4 * hiddenSize, hiddenSize])
        if bias {
            self._bias.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])
        } else {
            self._bias.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, hidden: MLXArray? = nil, cell: MLXArray? = nil) -> (MLXArray, MLXArray) {
        var x = x

        if let bias {
            x = addMM(bias, x, wx.T)
        } else {
            x = matmul(x, wx.T)
        }

        var currentHidden: MLXArray? = hidden
        var currentCell: MLXArray? = cell
        var allHidden = [MLXArray]()
        var allCell = [MLXArray]()

        for index in 0..<x.dim(-2) {
            var ifgo = x[.ellipsis, index, 0...]
            if let h = currentHidden {
                ifgo = addMM(ifgo, h, wh.T)
            }

            let pieces = split(ifgo, parts: 4, axis: -1)

            let i = sigmoid(pieces[0])
            let f = sigmoid(pieces[1])
            let g = tanh(pieces[2])
            let o = sigmoid(pieces[3])

            let newCell: MLXArray
            if let c = currentCell {
                newCell = f * c + i * g
            } else {
                newCell = i * g
            }
            currentCell = newCell

            let newHidden = o * tanh(newCell)
            currentHidden = newHidden

            allCell.append(newCell)
            allHidden.append(newHidden)
        }

        return (
            stacked(allHidden, axis: -2),
            stacked(allCell, axis: -2)
        )
    }

    func setWeights(wx: MLXArray, wh: MLXArray, bias: MLXArray?) {
        self._wx.wrappedValue = wx
        self._wh.wrappedValue = wh
        self._bias.wrappedValue = bias
    }
}

/// Pre-computed STFT layer using basis matrix from weights
final class STFT: Module {
    @ParameterInfo(key: "forward_basis_buffer") var forwardBasis: MLXArray

    init(forwardBasis: MLXArray) {
        self._forwardBasis.wrappedValue = forwardBasis
        super.init()
    }

    func callAsFunction(_ audio: MLXArray) -> MLXArray {
        // audio: [batch, 512] - 512 samples at 16kHz
        // forwardBasis: [258, 1, 256] - pre-computed STFT basis
        // Output: magnitude spectrogram [batch, frames, 129]

        var x = audio
        if x.ndim == 1 {
            x = x.expandedDimensions(axis: 0)
        }
        // x shape: [batch, samples]

        // For MLX conv1d, we need input as [batch, length, channels]
        // and weight as [out_channels, kernel_size, in_channels]
        // Reshape audio: [batch, samples, 1]
        x = x.expandedDimensions(axis: 2)

        // forwardBasis is [258, 1, 256] which is [out_channels, in_channels, kernel]
        // MLX expects [out_channels, kernel_size, in_channels]
        // So we need to transpose to [258, 256, 1]
        let kernel = forwardBasis.transposed(0, 2, 1)

        // Apply conv1d with stride=128 (hop length), padding to center
        let output = conv1d(x, kernel, stride: 128, padding: 128)
        // output shape: [batch, frames, 258]

        // Split into real and imaginary parts
        let nFreqs = 129  // 258 / 2
        let real = output[0..., 0..., ..<nFreqs]
        let imag = output[0..., 0..., nFreqs...]

        // Return magnitude spectrogram [batch, frames, 129]
        let magnitude = sqrt(real * real + imag * imag)
        return magnitude
    }

    func setForwardBasis(_ basis: MLXArray) {
        self._forwardBasis.wrappedValue = basis
    }
}

/// Silero VAD neural network model (stateless)
///
/// Architecture:
/// Audio (512 samples) -> STFT -> Encoder (4x Conv1d+ReLU) -> LSTM -> Output Conv1d -> Sigmoid
///
/// The model is stateless - LSTM state is passed in and out for streaming inference.
public final class SileroVADModel: Module {
    @ModuleInfo(key: "stft") var stft: STFT
    @ModuleInfo(key: "encoder") var encoder: [VADConv1d]
    @ModuleInfo(key: "decoder") var lstm: VADLSTM
    @ModuleInfo(key: "output") var outputConv: VADConv1d

    public override init() {
        self._stft.wrappedValue = STFT(forwardBasis: MLXArray.zeros([258, 1, 256]))
        self._encoder.wrappedValue = [
            VADConv1d(inputChannels: 129, outputChannels: 128, kernelSize: 3, padding: 1),
            VADConv1d(inputChannels: 128, outputChannels: 64, kernelSize: 3, padding: 1),
            VADConv1d(inputChannels: 64, outputChannels: 64, kernelSize: 3, padding: 1),
            VADConv1d(inputChannels: 64, outputChannels: 128, kernelSize: 3, padding: 1),
        ]
        self._lstm.wrappedValue = VADLSTM(inputSize: 128, hiddenSize: 128)
        self._outputConv.wrappedValue = VADConv1d(inputChannels: 128, outputChannels: 1, kernelSize: 1, bias: false)
        super.init()
    }

    /// Forward pass with external state management
    /// - Parameters:
    ///   - audio: Audio chunk [batch, 512] or [512]
    ///   - state: LSTM state (hidden, cell)
    /// - Returns: (probability [0-1], newState)
    public func callAsFunction(_ audio: MLXArray, state: VADState) -> (MLXArray, VADState) {
        var x = audio
        if x.ndim == 1 {
            x = x.expandedDimensions(axis: 0)
        }

        // 1. STFT: audio -> spectrogram [batch, frames, 129]
        x = stft(x)

        // 2. Encoder: 4x (Conv1d -> ReLU)
        // Conv1d in MLX expects [batch, length, channels]
        for conv in encoder {
            x = relu(conv(x))
        }
        // x shape: [batch, frames, 128]

        // 3. LSTM with state
        // LSTM expects [batch, sequence, features]
        let (allHidden, allCell) = lstm(x, hidden: state.hidden, cell: state.cell)

        // Get last hidden and cell states for next iteration
        // allHidden/allCell shape: [batch, sequence, hidden]
        let seqLen = allHidden.shape[1]
        let newHidden = allHidden[0..., (seqLen - 1)..<seqLen, 0...].squeezed(axis: 1)
        let newCell = allCell[0..., (seqLen - 1)..<seqLen, 0...].squeezed(axis: 1)

        // 4. Output conv
        // output shape: [batch, frames, 128] -> [batch, frames, 1]
        var output = outputConv(allHidden)

        // 5. Sigmoid -> probability
        output = sigmoid(output)

        // Take the last frame's probability
        // output shape: [batch, frames, 1]
        let lastFrame = output.shape[1] - 1
        let probability = output[0, lastFrame, 0]

        // Evaluate state tensors to prevent computation graph memory buildup
        eval(newHidden, newCell)
        let newState = VADState(hidden: newHidden, cell: newCell)
        return (probability, newState)
    }

    /// Load weights from safetensors file
    public static func load(from url: URL) throws -> SileroVADModel {
        let model = SileroVADModel()

        guard FileManager.default.fileExists(atPath: url.path) else {
            throw VADError.weightsNotFound(path: url.path)
        }

        let weights: [String: MLXArray]
        do {
            weights = try loadArrays(url: url)
        } catch {
            throw VADError.weightsCorrupted(reason: error.localizedDescription)
        }

        try model.loadWeights(weights)
        return model
    }

    private func loadWeights(_ weights: [String: MLXArray]) throws {
        // Load STFT basis
        guard let stftBasis = weights["stft.forward_basis_buffer"] else {
            throw VADError.weightsCorrupted(reason: "Missing stft.forward_basis_buffer")
        }
        stft.setForwardBasis(stftBasis)

        // Load encoder conv weights
        // PyTorch Conv1d weight shape: [out_channels, in_channels, kernel_size]
        // MLX Conv1d weight shape: [out_channels, kernel_size, in_channels]
        let encoderMapping = [
            (0, "encoder.0.reparam_conv"),
            (1, "encoder.1.reparam_conv"),
            (2, "encoder.2.reparam_conv"),
            (3, "encoder.3.reparam_conv"),
        ]

        for (idx, prefix) in encoderMapping {
            guard let weight = weights["\(prefix).weight"] else {
                throw VADError.weightsCorrupted(reason: "Missing \(prefix).weight")
            }
            guard let bias = weights["\(prefix).bias"] else {
                throw VADError.weightsCorrupted(reason: "Missing \(prefix).bias")
            }
            // Transpose weight from [out, in, kernel] to [out, kernel, in]
            encoder[idx].setWeights(weight: weight.transposed(0, 2, 1), bias: bias)
        }

        // Load LSTM weights
        // PyTorch LSTM has separate weight_ih, weight_hh, bias_ih, bias_hh
        guard let weightIh = weights["decoder.rnn.weight_ih"],
              let weightHh = weights["decoder.rnn.weight_hh"],
              let biasIh = weights["decoder.rnn.bias_ih"],
              let biasHh = weights["decoder.rnn.bias_hh"]
        else {
            throw VADError.weightsCorrupted(reason: "Missing LSTM weights")
        }

        // Combine biases (PyTorch LSTM adds both biases)
        lstm.setWeights(wx: weightIh, wh: weightHh, bias: biasIh + biasHh)

        // Load output conv weights
        guard let outputWeight = weights["decoder.decoder.2.weight"] else {
            throw VADError.weightsCorrupted(reason: "Missing decoder.decoder.2.weight")
        }
        // Output weight is [1, 128, 1] in PyTorch (out, in, kernel)
        // Transpose to [1, 1, 128] for MLX
        outputConv.setWeights(weight: outputWeight.transposed(0, 2, 1))
    }
}
