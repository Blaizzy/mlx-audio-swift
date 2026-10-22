import Foundation
import MLX
import MLXAudioCore

public enum GraniteSpeech5CTCFrontend {
    /// Replicate-padded time deltas, matching the reference's torchaudio recipe.
    public static func deltas(_ features: MLXArray, windowLength: Int = 3) -> MLXArray {
        precondition(
            features.ndim == 2 && features.dim(0) > 0 && windowLength >= 3 && windowLength % 2 == 1,
            "Deltas require nonempty [time,features] and an odd window of at least 3")
        let half = (windowLength - 1) / 2, t = features.dim(0), d = features.dim(1)
        let data = concatenated([broadcast(features[0..<1], to: [half,d]), features,
            broadcast(features[(t - 1)..<t], to: [half,d])], axis: 0)
        var result = zeros(like: features)
        var denominator = 0
        for offset in 1...half {
            result = result + Float(offset) * (data[(half + offset)..<(half + offset + t)]
                - data[(half - offset)..<(half - offset + t)])
            denominator += 2 * offset * offset
        }
        return result / Float(denominator)
    }

    /// HTK filters calculated in Double before a single float32 cast, as Python precise=True.
    static func filters(_ count: Int) -> MLXArray {
        let maximum = 2595.0 * log10(1 + 8000.0 / 700)
        let points = (0..<(count + 2)).map { 700.0 * (pow(10.0, Double($0) * maximum / Double(count + 1) / 2595) - 1) }
        var values = [Float](repeating: 0, count: 257 * count)
        for f in 0..<257 {
            let frequency = Double(f) * 16000 / 512
            for m in 0..<count {
                let down = (frequency - points[m]) / (points[m + 1] - points[m])
                let up = (points[m + 2] - frequency) / (points[m + 2] - points[m + 1])
                values[f * count + m] = Float(max(0, min(down, up)))
            }
        }
        return MLXArray(values).reshaped(257,count)
    }

    /// Mono 16 kHz waveform to pair-stacked log-mel and delta features [time,4*mels].
    public static func features(_ waveform: MLXArray, numMelBins: Int = 80) -> MLXArray {
        precondition(
            waveform.ndim == 1 && waveform.size >= 320 && numMelBins > 0,
            "Granite frontend requires mono audio containing at least 320 samples")
        var audio = waveform.asType(.float32)
        let melFrames = audio.size / 160, frames = 2 * ((melFrames + 1) / 2)
        let required = (frames - 1) * 160 + 1
        if audio.size < required { audio = padded(audio, widths: [.init((0,required - audio.size))]) }
        let window = padded(hanningWindow(size: 400, periodic: true), widths: [.init((56,56))])
        let spectrum = stft(audio: audio, window: window, nFft: 512, hopLength: 160)[0..<frames]
        let mel = matmul(abs(spectrum).square(), filters(numMelBins))
        var logmel = log10(maximum(mel, MLXArray(Float(1e-10))))
        logmel = maximum(logmel, logmel.max() - 8) / 4 + 1
        return concatenated([logmel, deltas(logmel)], axis: -1).reshaped(-1,4 * numMelBins)
    }
}

/// Decode-only vocabulary support for the checkpoint's SentencePiece-derived BPE.
/// No text encoding or Python processor is required by greedy CTC inference.
public final class GraniteSpeech5CTCTokenizer {
    private let vocabulary: [Int:String]
    private let special: Set<Int>
    public init(directory: URL, vocabSize: Int) throws {
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("tokenizer.json")))
        guard let json = object as? [String:Any], let model = json["model"] as? [String:Any],
              let vocabulary = model["vocab"] as? [String:Int],
              vocabulary.count == vocabSize, Set(vocabulary.values) == Set(0..<vocabSize) else {
            throw STTError.invalidInput("Granite tokenizer must contain a complete contiguous BPE vocabulary")
        }
        self.vocabulary = Dictionary(uniqueKeysWithValues: vocabulary.map { ($0.value,$0.key) })
        let added = json["added_tokens"] as? [[String:Any]] ?? []
        special = Set(added.compactMap { ($0["special"] as? Bool == true) ? $0["id"] as? Int : nil })
    }
    public func decode(_ tokens: [Int]) -> String {
        var result = "", bytes: [UInt8] = []
        func flush() {
            if !bytes.isEmpty { result += String(decoding: bytes, as: UTF8.self); bytes.removeAll(keepingCapacity: true) }
        }
        for id in tokens where !special.contains(id) {
            let piece = vocabulary[id] ?? ""
            if piece.hasPrefix("<0x"), piece.hasSuffix(">"), let byte = UInt8(piece.dropFirst(3).dropLast(), radix: 16) {
                bytes.append(byte)
            } else { flush(); result += piece.replacingOccurrences(of: "▁", with: " ") }
        }
        flush()
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
