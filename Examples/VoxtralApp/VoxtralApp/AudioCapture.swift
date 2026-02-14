@preconcurrency import AVFoundation

protocol AudioCaptureDelegate: AnyObject {
    func audioCapture(_ capture: AudioCapture, didReceiveBuffer buffer: [Float])
}

final class AudioCapture {
    weak var delegate: AudioCaptureDelegate?

    private let engine = AVAudioEngine()
    private let targetSampleRate: Double = 16000
    private var converter: AVAudioConverter?

    func start() {
        AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
            guard granted, let self else { return }
            do {
                try self.setupAndStart()
            } catch {
                print("AudioCapture failed to start: \(error)")
            }
        }
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        converter = nil
    }

    private func setupAndStart() throws {
        let inputNode = engine.inputNode
        let hwFormat = inputNode.outputFormat(forBus: 0)

        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        )!

        let needsConversion = hwFormat.sampleRate != targetSampleRate || hwFormat.channelCount != 1

        if needsConversion {
            converter = AVAudioConverter(from: hwFormat, to: targetFormat)
        }

        // Use hardware buffer size — tap must match inputNode's output format
        let bufferSize: AVAudioFrameCount = 4096
        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: hwFormat) {
            [weak self] buffer, _ in
            guard let self else { return }

            if let converter = self.converter {
                let frameCapacity = AVAudioFrameCount(
                    Double(buffer.frameLength) * self.targetSampleRate / hwFormat.sampleRate
                )
                guard
                    let convertedBuffer = AVAudioPCMBuffer(
                        pcmFormat: targetFormat, frameCapacity: frameCapacity)
                else { return }

                var error: NSError?
                var consumed = false
                converter.convert(to: convertedBuffer, error: &error) { _, status in
                    if consumed {
                        status.pointee = .noDataNow
                        return nil
                    }
                    consumed = true
                    status.pointee = .haveData
                    return buffer
                }
                if error == nil, convertedBuffer.frameLength > 0 {
                    self.deliver(convertedBuffer)
                }
            } else {
                self.deliver(buffer)
            }
        }

        engine.prepare()
        try engine.start()
    }

    private func deliver(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }
        let frames = Int(buffer.frameLength)
        let samples = Array(UnsafeBufferPointer(start: channelData[0], count: frames))
        delegate?.audioCapture(self, didReceiveBuffer: samples)
    }
}
