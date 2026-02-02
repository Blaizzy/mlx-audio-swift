import Foundation

/// Options for model loading behavior
public struct ModelLoadingOptions: Sendable {
    public var quantization: WhisperQuantization
    public var loadInBackground: Bool
    public var fallbackToFloat16: Bool

    public init(
        quantization: WhisperQuantization = .float16,
        loadInBackground: Bool = false,
        fallbackToFloat16: Bool = true
    ) {
        self.quantization = quantization
        self.loadInBackground = loadInBackground
        self.fallbackToFloat16 = fallbackToFloat16
    }

    // MARK: - Presets

    /// Default: float16, blocking, with fallback
    public static let `default` = ModelLoadingOptions(
        quantization: .float16,
        loadInBackground: false,
        fallbackToFloat16: true
    )

    /// Fast: int4, background loading, with fallback
    public static let fast = ModelLoadingOptions(
        quantization: .int4,
        loadInBackground: true,
        fallbackToFloat16: true
    )

    /// Fast but blocking: int4, wait for eval, with fallback
    public static let fastBlocking = ModelLoadingOptions(
        quantization: .int4,
        loadInBackground: false,
        fallbackToFloat16: true
    )

    /// Strict: int4, fail if unavailable (no fallback)
    public static let strict = ModelLoadingOptions(
        quantization: .int4,
        loadInBackground: false,
        fallbackToFloat16: false
    )
}
