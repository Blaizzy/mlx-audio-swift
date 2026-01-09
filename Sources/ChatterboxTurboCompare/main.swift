import AVFoundation
import Foundation
import MLX

import MLXAudioTTS

@main
struct ChatterboxTurboCompare {
    static func main() async throws {
        var repo = "mlx-community/Chatterbox-Turbo-TTS-4bit"
        var text = "Quick quality check. Does this sound natural?"
        var outPath: String?
        var tokensOutPath: String?
        var melOutPath: String?
        var dumpParamKey: String?
        var dumpParamOutPath: String?
        var seed: UInt64 = 0

        var maxTokens = 200
        var repetitionPenalty: Float = 1.2
        var topP: Float = 0.95
        var temperature: Float = 0.8
        var topK = 1000

        var index = 1
        let args = CommandLine.arguments
        while index < args.count {
            let arg = args[index]
            switch arg {
            case "--repo":
                index += 1
                repo = args[safe: index] ?? repo
            case "--text":
                index += 1
                text = args[safe: index] ?? text
            case "--out":
                index += 1
                outPath = args[safe: index]
            case "--seed":
                index += 1
                if let value = args[safe: index], let parsed = UInt64(value) {
                    seed = parsed
                }
            case "--tokensOut":
                index += 1
                tokensOutPath = args[safe: index]
            case "--melOut":
                index += 1
                melOutPath = args[safe: index]
            case "--dumpParamKey":
                index += 1
                dumpParamKey = args[safe: index]
            case "--dumpParamOut":
                index += 1
                dumpParamOutPath = args[safe: index]
            case "--maxTokens":
                index += 1
                if let value = args[safe: index], let parsed = Int(value) {
                    maxTokens = parsed
                }
            case "--repetitionPenalty":
                index += 1
                if let value = args[safe: index], let parsed = Float(value) {
                    repetitionPenalty = parsed
                }
            case "--topP":
                index += 1
                if let value = args[safe: index], let parsed = Float(value) {
                    topP = parsed
                }
            case "--temperature":
                index += 1
                if let value = args[safe: index], let parsed = Float(value) {
                    temperature = parsed
                }
            case "--topK":
                index += 1
                if let value = args[safe: index], let parsed = Int(value) {
                    topK = parsed
                }
            case "--help", "-h":
                printUsageAndExit(exitCode: 0)
            default:
                print("Unknown argument: \(arg)")
                printUsageAndExit(exitCode: 2)
            }
            index += 1
        }

        if outPath == nil, dumpParamKey == nil {
            printUsageAndExit(exitCode: 2)
        }

        let model = try await ChatterboxTurboTTS.fromPretrained(repo)

        if let dumpParamKey, let dumpParamOutPath {
            let params = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) })
            guard let value = params[dumpParamKey] else {
                print("Parameter not found: \(dumpParamKey)")
                exit(2)
            }

            let outURL = URL(fileURLWithPath: dumpParamOutPath)
            try writeTensor(value, to: outURL)
            print("Wrote \(outURL.path)")
            print("param \(dumpParamKey) shape \(value.shape) dtype \(value.dtype)")
            exit(0)
        }

        guard let outPath else {
            printUsageAndExit(exitCode: 2)
        }

        let state = MLXRandom.RandomState(seed: seed)
        let (speechTokens, audio) = try withRandomState(state) {
            let speechTokens = try model.generateSpeechTokens(
                text: text,
                repetitionPenalty: repetitionPenalty,
                topP: topP,
                temperature: temperature,
                topK: topK,
                splitPattern: nil,
                maxTokens: maxTokens
            )
            let audio = try model.generateWav(speechTokens: speechTokens, nCfmTimesteps: 2)
            return (speechTokens, audio)
        }

        let url = URL(fileURLWithPath: outPath)
        try writeWav(samples: audio.asArray(Float.self), sampleRate: Double(model.sampleRate), to: url)

        if let tokensOutPath {
            let tokensURL = URL(fileURLWithPath: tokensOutPath)
            try writeTokens(speechTokens, to: tokensURL)
            print("Wrote \(tokensURL.path)")
        }

        if let melOutPath {
            let melURL = URL(fileURLWithPath: melOutPath)
            let melState = MLXRandom.RandomState(seed: seed)
            let mel = try withRandomState(melState) {
                let speechTokens = try model.generateSpeechTokens(
                    text: text,
                    repetitionPenalty: repetitionPenalty,
                    topP: topP,
                    temperature: temperature,
                    topK: topK,
                    splitPattern: nil,
                    maxTokens: maxTokens
                )
                return try model.generateMel(speechTokens: speechTokens, nCfmTimesteps: 2)
            }
            try writeMel(mel, to: melURL)
            print("Wrote \(melURL.path)")
        }

        print("Wrote \(url.path)")
        print("sample_rate \(model.sampleRate) samples \(audio.shape[0])")
    }
}

private extension Array where Element == String {
    subscript(safe index: Int) -> String? {
        guard index >= 0, index < count else { return nil }
        return self[index]
    }
}

private func printUsageAndExit(exitCode: Int32) -> Never {
    print(
        """
        Usage: ChatterboxTurboCompare --out <path> [options]

        Options:
          --repo <id>              HF repo (default: mlx-community/Chatterbox-Turbo-TTS-4bit)
          --text <text>            Prompt text
          --seed <n>               Random seed (default: 0)
          --tokensOut <path>       Write generated speech tokens to a text file (space-separated)
          --melOut <path>          Write generated mels as raw float32 with 2x int32 header (nMels, frames)
          --dumpParamKey <key>     Dump a model parameter by flattened key
          --dumpParamOut <path>    Output path for --dumpParamKey (raw float32 with int32 header)
          --maxTokens <n>          Max speech tokens (default: 200)
          --repetitionPenalty <f>  (default: 1.2)
          --topP <f>               (default: 0.95)
          --temperature <f>        (default: 0.8)
          --topK <n>               (default: 1000)
        """
    )
    exit(exitCode)
}

private func writeWav(samples: [Float], sampleRate: Double, to url: URL) throws {
    let directory = url.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    if FileManager.default.fileExists(atPath: url.path) {
        try FileManager.default.removeItem(at: url)
    }

    let frameCount = AVAudioFrameCount(samples.count)
    guard
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ),
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)
    else {
        throw NSError(domain: "ChatterboxTurboCompare", code: 1)
    }
    buffer.frameLength = frameCount

    guard let channelData = buffer.floatChannelData?[0] else {
        throw NSError(domain: "ChatterboxTurboCompare", code: 2)
    }
    samples.withUnsafeBufferPointer { src in
        channelData.update(from: src.baseAddress!, count: src.count)
    }

    let audioFile = try AVAudioFile(
        forWriting: url,
        settings: format.settings,
        commonFormat: format.commonFormat,
        interleaved: format.isInterleaved
    )
    try audioFile.write(from: buffer)
}

private func writeTokens(_ tokens: MLXArray, to url: URL) throws {
    let directory = url.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    if FileManager.default.fileExists(atPath: url.path) {
        try FileManager.default.removeItem(at: url)
    }

    let flat = tokens.reshaped([-1]).asArray(Int32.self)
    let contents = flat.map(String.init).joined(separator: " ")
    try contents.write(to: url, atomically: true, encoding: .utf8)
}

private func writeMel(_ mel: MLXArray, to url: URL) throws {
    let directory = url.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    if FileManager.default.fileExists(atPath: url.path) {
        try FileManager.default.removeItem(at: url)
    }

    let mel2d = mel.ndim == 3 ? mel.squeezed(axis: 0) : mel
    guard mel2d.ndim == 2 else {
        throw NSError(domain: "ChatterboxTurboCompare", code: 3)
    }

    let nMels = Int32(mel2d.dim(0))
    let frames = Int32(mel2d.dim(1))
    let values = mel2d.asArray(Float.self)

    var data = Data()
    var nMelsLe = nMels.littleEndian
    var framesLe = frames.littleEndian
    withUnsafeBytes(of: &nMelsLe) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: &framesLe) { data.append(contentsOf: $0) }
    values.withUnsafeBytes { data.append(contentsOf: $0) }

    try data.write(to: url)
}

private func writeTensor(_ array: MLXArray, to url: URL) throws {
    let directory = url.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    if FileManager.default.fileExists(atPath: url.path) {
        try FileManager.default.removeItem(at: url)
    }

    let floatArray = array.asType(.float32)
    let values = floatArray.asArray(Float.self)

    var data = Data()
    var ndimLe = Int32(floatArray.ndim).littleEndian
    withUnsafeBytes(of: &ndimLe) { data.append(contentsOf: $0) }

    for dim in floatArray.shape {
        var dimLe = Int32(dim).littleEndian
        withUnsafeBytes(of: &dimLe) { data.append(contentsOf: $0) }
    }

    values.withUnsafeBytes { data.append(contentsOf: $0) }
    try data.write(to: url)
}
