import Foundation
import MLX
import MLXNN
import MLXAudioCodecs

// MARK: - Classifier Components

class DNNLinear: Module {
    @ModuleInfo var w: Linear

    init(inputDim: Int, outputDim: Int) {
        _w.wrappedValue = Linear(inputDim, outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { w(x) }
}

class DNNBlock: Module {
    @ModuleInfo var linear: DNNLinear
    @ModuleInfo var norm: BatchNorm

    init(inputDim: Int, outputDim: Int) {
        _linear.wrappedValue = DNNLinear(inputDim: inputDim, outputDim: outputDim)
        _norm.wrappedValue = BatchNorm(featureCount: outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        norm(leakyRelu(linear(x)))
    }
}

class DNN: Module {
    @ModuleInfo(key: "block_0") var block0: DNNBlock

    init(inputDim: Int, outputDim: Int) {
        _block0.wrappedValue = DNNBlock(inputDim: inputDim, outputDim: outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { block0(x) }
}

class ClassifierLinear: Module {
    @ModuleInfo var w: Linear

    init(inputDim: Int, outputDim: Int) {
        _w.wrappedValue = Linear(inputDim, outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { w(x) }
}

class EcapaClassifier: Module {
    @ModuleInfo var norm: BatchNorm
    @ModuleInfo(key: "DNN") var dnn: DNN
    @ModuleInfo var out: ClassifierLinear

    init(config: EcapaTdnnConfig) {
        _norm.wrappedValue = BatchNorm(featureCount: config.embeddingDim)
        _dnn.wrappedValue = .init(inputDim: config.embeddingDim, outputDim: config.classifierHiddenDim)
        _out.wrappedValue = ClassifierLinear(inputDim: config.classifierHiddenDim, outputDim: config.numClasses)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        if out.ndim == 3 {
            if out.dim(1) == 1 {
                out = out.squeezed(axis: 1)
            } else if out.dim(2) == 1 {
                out = out.squeezed(axis: 2)
            }
        }
        out = leakyRelu(out)
        out = norm(out)
        out = dnn(out)
        out = self.out(out)
        return logSoftmax(out, axis: -1)
    }
}
