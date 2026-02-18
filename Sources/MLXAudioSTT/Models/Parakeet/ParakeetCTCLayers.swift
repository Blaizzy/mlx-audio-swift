import Foundation
import MLX
import MLXNN

final class ParakeetConvASRDecoder: Module {
    @ModuleInfo(key: "decoder_layers") var decoder: Conv1d
    let temperature: Float

    init(args: ParakeetConvASRDecoderConfig, temperature: Float = 1.0) {
        let classCount = (args.numClasses <= 0 ? args.vocabulary.count : args.numClasses) + 1
        guard let featIn = args.featIn else {
            fatalError("ParakeetConvASRDecoder requires featIn to be resolved before initialization")
        }
        self.temperature = temperature
        self._decoder.wrappedValue = Conv1d(
            inputChannels: featIn,
            outputChannels: classCount,
            kernelSize: 1,
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let logits = decoder(x) / temperature
        return logSoftmax(logits, axis: -1)
    }
}
