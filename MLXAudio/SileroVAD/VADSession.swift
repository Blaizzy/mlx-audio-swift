import Foundation
import MLX
import MLXNN

/// Thread-safe factory and convenience methods for Silero VAD.
/// Create once, use from any thread.
///
/// The session wraps a loaded VAD model and provides:
/// - Factory methods for loading models with bundled or custom weights
/// - Creation of independent iterators for streaming audio processing
/// - Batch processing via `getSpeechTimestamps`
///
/// Thread safety: The underlying model is safe to share because weights are loaded
/// once and never modified, forward pass only reads weights, and each iterator
/// maintains its own state.
public final class VADSession: @unchecked Sendable {
    public static let sampleRate: Int = VADAudioFormat.sampleRate

    private static let bundledWeightsBaseName = "silero_vad_16k"
    private static let bundledWeightsExtension = "safetensors"
    private static var bundledWeightsFilename: String {
        "\(bundledWeightsBaseName).\(bundledWeightsExtension)"
    }

    private let model: SileroVADModel

    /// Load model with bundled weights.
    /// - Returns: Configured VADSession ready for use
    /// - Throws: VADError if weights cannot be found or loaded
    public static func make() async throws -> VADSession {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let weightsURL = try findBundledWeights()
                    let model = try SileroVADModel.load(from: weightsURL)
                    continuation.resume(returning: VADSession(model: model))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Load model from custom weights path.
    /// - Parameter weightsURL: URL to safetensors weights file
    /// - Returns: Configured VADSession ready for use
    /// - Throws: VADError if weights cannot be loaded
    public static func make(weightsURL: URL) async throws -> VADSession {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let model = try SileroVADModel.load(from: weightsURL)
                    continuation.resume(returning: VADSession(model: model))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Create independent iterator for streaming audio processing.
    /// Each iterator maintains its own state and can be used independently.
    /// - Parameter config: VAD configuration for speech detection thresholds
    /// - Returns: New VADIterator instance
    public func makeIterator(config: VADConfig = .default) -> VADIterator {
        VADIterator(model: model, config: config)
    }

    /// Batch processing convenience for extracting speech segments from audio.
    /// - Parameters:
    ///   - audio: Complete audio signal as MLXArray
    ///   - config: VAD configuration for speech detection
    /// - Returns: Array of detected speech segments with start/end timestamps
    /// - Throws: VADError if processing fails
    public func getSpeechTimestamps(
        _ audio: MLXArray,
        config: VADConfig = .default
    ) throws -> [SpeechSegment] {
        try SileroVAD.getSpeechTimestamps(audio: audio, model: model, config: config)
    }

    internal init(model: SileroVADModel) {
        self.model = model
    }

    private static func findBundledWeights() throws -> URL {
        // Try Bundle.main first (app bundle)
        if let resourceURL = Bundle.main.resourceURL {
            let mainURL = resourceURL.appendingPathComponent(bundledWeightsFilename)
            if FileManager.default.fileExists(atPath: mainURL.path) {
                return mainURL
            }
        }

        // Try Bundle.module (Swift Package)
        #if SWIFT_PACKAGE
        let moduleURL = Bundle.module.url(forResource: bundledWeightsBaseName, withExtension: bundledWeightsExtension)
        if let url = moduleURL, FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        #endif

        // Try searching in bundle resources
        if let url = Bundle.main.url(forResource: bundledWeightsBaseName, withExtension: bundledWeightsExtension) {
            return url
        }

        throw VADError.weightsNotFound(path: bundledWeightsFilename)
    }
}
