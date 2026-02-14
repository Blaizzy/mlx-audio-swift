import Foundation

protocol EnergyVADDelegate: AnyObject {
    func vadDidDetectSpeechStart()
    func vadDidDetectSpeechEnd(audio: [Float])
}

final class EnergyVAD {
    enum State {
        case idle
        case speech
        case hangtime
    }

    weak var delegate: EnergyVADDelegate?

    var energyThreshold: Float = 0.01
    var hangTime: TimeInterval = 1.5

    private(set) var state: State = .idle
    private var accumulatedAudio: [Float] = []
    private var hangtimeStart: Date?

    func processBuffer(_ buffer: [Float]) {
        guard !buffer.isEmpty else { return }

        let sumOfSquares = buffer.reduce(Float(0)) { $0 + $1 * $1 }
        let rms = sqrt(sumOfSquares / Float(buffer.count))
        let isSpeech = rms > energyThreshold

        switch state {
        case .idle:
            if isSpeech {
                state = .speech
                accumulatedAudio.removeAll()
                accumulatedAudio.append(contentsOf: buffer)
                delegate?.vadDidDetectSpeechStart()
            }

        case .speech:
            accumulatedAudio.append(contentsOf: buffer)
            if !isSpeech {
                state = .hangtime
                hangtimeStart = Date()
            }

        case .hangtime:
            accumulatedAudio.append(contentsOf: buffer)
            if isSpeech {
                state = .speech
                hangtimeStart = nil
            } else if let start = hangtimeStart, Date().timeIntervalSince(start) >= hangTime {
                let audio = accumulatedAudio
                accumulatedAudio.removeAll()
                hangtimeStart = nil
                state = .idle
                delegate?.vadDidDetectSpeechEnd(audio: audio)
            }
        }
    }

    func reset() {
        state = .idle
        accumulatedAudio.removeAll()
        hangtimeStart = nil
    }
}
