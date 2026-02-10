//
//  WiredMemoryManager.swift
//  MLXAudioCore
//
//  Utilities for pinning model weights in physical memory using MLX's wired memory API.
//  Wired memory prevents macOS from paging out GPU buffers under memory pressure,
//  reducing latency spikes during real-time TTS/STT inference.
//
//  Requires macOS 15+ / Metal 3 (Apple Silicon). Degrades gracefully on older systems.
//

import Foundation
@preconcurrency import MLX

// MARK: - WiredMemoryManager

/// Manages wired (pinned) memory for MLX model weights.
///
/// When model weights are loaded into GPU memory, macOS may page them out under memory
/// pressure. For real-time audio applications (TTS/STT), this causes unpredictable latency
/// spikes as weights must be paged back in. Wired memory pins buffers in physical RAM,
/// guaranteeing they remain resident.
///
/// This manager wraps MLX's `Memory.withWiredLimit()` API, which uses Metal residency sets
/// (macOS 15+, Metal 3) to keep allocated buffers resident up to the specified limit.
///
/// ## Usage
///
/// ### Pin memory for a generation block
/// ```swift
/// let audio = try await WiredMemoryManager.withPinnedMemory {
///     try await model.generate(text: "Hello", voice: "A clear voice", language: "en")
/// }
/// ```
///
/// ### Pin with explicit byte limit
/// ```swift
/// let audio = try await WiredMemoryManager.withPinnedMemory(limitBytes: 4_000_000_000) {
///     try await model.generate(text: "Hello", voice: "A clear voice", language: "en")
/// }
/// ```
///
/// ### Query system limits
/// ```swift
/// let info = WiredMemoryManager.systemInfo()
/// print("Max wirable: \(info.maxWirableBytes / 1_000_000)MB")
/// ```
public enum WiredMemoryManager {

    // MARK: - System Info

    /// Information about system wired memory capabilities.
    public struct SystemInfo: Sendable {
        /// Maximum recommended working set size reported by the GPU device, in bytes.
        /// This is the upper bound for the wired memory limit.
        public let maxRecommendedWorkingSetSize: UInt64

        /// Total physical memory on the system, in bytes.
        public let totalMemoryBytes: Int

        /// Maximum bytes that can be safely wired without exceeding system limits.
        /// This is `maxRecommendedWorkingSetSize` on supported systems, or 0 if
        /// wired memory is not available.
        public let maxWirableBytes: Int

        /// GPU architecture name (e.g., "Apple M4 Pro").
        public let architecture: String

        /// Whether the system supports Metal residency sets (macOS 15+, Metal 3).
        /// Wired memory has no effect on systems without this support.
        public let supportsWiredMemory: Bool
    }

    /// Query system wired memory capabilities.
    ///
    /// Returns information about the GPU device and the maximum amount of memory
    /// that can be wired. Use this to make informed decisions about wired limits.
    ///
    /// - Returns: A ``SystemInfo`` describing the system's wired memory capabilities.
    public static func systemInfo() -> SystemInfo {
        let deviceInfo = GPU.deviceInfo()
        let maxWorkingSet = deviceInfo.maxRecommendedWorkingSetSize
        // Metal residency sets require macOS 15+ and Metal 3 (Apple Silicon).
        // maxRecommendedWorkingSetSize == 0 indicates no GPU or unsupported device.
        let supportsWired = maxWorkingSet > 0
        return SystemInfo(
            maxRecommendedWorkingSetSize: maxWorkingSet,
            totalMemoryBytes: deviceInfo.memorySize,
            maxWirableBytes: supportsWired ? Int(maxWorkingSet) : 0,
            architecture: deviceInfo.architecture,
            supportsWiredMemory: supportsWired
        )
    }

    // MARK: - Wired Memory Execution (Synchronous)

    /// Execute a block with model weights pinned in physical memory.
    ///
    /// This sets the MLX wired memory limit for the duration of the block, causing
    /// all MLX GPU allocations (including model weights and intermediate buffers)
    /// to be kept resident in physical RAM up to the specified limit.
    ///
    /// If `limitBytes` exceeds the system's maximum recommended working set size,
    /// the limit is clamped and a warning is printed. If the system does not support
    /// wired memory at all, the block executes normally without pinning.
    ///
    /// - Parameters:
    ///   - limitBytes: Maximum bytes to wire. Pass `nil` to use the system's maximum
    ///     recommended working set size (the safest default for pinning all model weights).
    ///   - body: The block to execute with wired memory enabled.
    /// - Returns: The return value of `body`.
    /// - Throws: Rethrows any error from `body`.
    public static func withPinnedMemory<R>(
        limitBytes: Int? = nil,
        _ body: () throws -> R
    ) rethrows -> R {
        let info = systemInfo()

        guard info.supportsWiredMemory else {
            print("[WiredMemoryManager] Wired memory not available on this system (\(info.architecture)). Continuing without pinning.")
            return try body()
        }

        let resolvedLimit = resolveLimit(requested: limitBytes, maxWirable: info.maxWirableBytes)

        print("[WiredMemoryManager] Pinning up to \(resolvedLimit / 1_000_000)MB of GPU memory (max: \(info.maxWirableBytes / 1_000_000)MB)")

        let result = try Memory.withWiredLimit(resolvedLimit) {
            try body()
        }

        print("[WiredMemoryManager] Wired memory released.")
        return result
    }

    // MARK: - Wired Memory Execution (Async)

    /// Execute an async block with model weights pinned in physical memory.
    ///
    /// This is the async variant of ``withPinnedMemory(limitBytes:_:)-6g9p3``, suitable
    /// for use in model generation methods that are `async`.
    ///
    /// - Parameters:
    ///   - limitBytes: Maximum bytes to wire. Pass `nil` to use the system's maximum.
    ///   - body: The async block to execute with wired memory enabled.
    /// - Returns: The return value of `body`.
    /// - Throws: Rethrows any error from `body`.
    public static func withPinnedMemory<R>(
        limitBytes: Int? = nil,
        _ body: () async throws -> R
    ) async rethrows -> R {
        let info = systemInfo()

        guard info.supportsWiredMemory else {
            print("[WiredMemoryManager] Wired memory not available on this system (\(info.architecture)). Continuing without pinning.")
            return try await body()
        }

        let resolvedLimit = resolveLimit(requested: limitBytes, maxWirable: info.maxWirableBytes)

        print("[WiredMemoryManager] Pinning up to \(resolvedLimit / 1_000_000)MB of GPU memory (max: \(info.maxWirableBytes / 1_000_000)MB)")

        let result = try await Memory.withWiredLimit(resolvedLimit) {
            try await body()
        }

        print("[WiredMemoryManager] Wired memory released.")
        return result
    }

    // MARK: - Convenience: Pin Based on Active Memory

    /// Execute a block with wired memory sized to cover the current active memory footprint,
    /// plus a growth headroom factor.
    ///
    /// This is useful after loading a model: call this to pin approximately the model's
    /// weight size plus room for inference buffers.
    ///
    /// - Parameters:
    ///   - headroomFactor: Multiplier on the current active memory to allow for inference
    ///     buffers. Defaults to 1.5 (50% headroom). For example, if the model uses 2GB,
    ///     the wired limit will be set to 3GB.
    ///   - body: The block to execute.
    /// - Returns: The return value of `body`.
    /// - Throws: Rethrows any error from `body`.
    public static func withPinnedMemoryForCurrentModel<R>(
        headroomFactor: Double = 1.5,
        _ body: () async throws -> R
    ) async rethrows -> R {
        let activeBytes = Memory.activeMemory
        let desiredLimit = Int(Double(activeBytes) * headroomFactor)
        return try await withPinnedMemory(limitBytes: desiredLimit, body)
    }

    // MARK: - Internal

    /// Resolve the wired memory limit, clamping to the system maximum with a warning
    /// if the requested limit exceeds it.
    private static func resolveLimit(requested: Int?, maxWirable: Int) -> Int {
        guard let requested = requested else {
            // Default: use the full recommended working set size
            return maxWirable
        }

        if requested > maxWirable {
            print("[WiredMemoryManager] WARNING: Requested wired limit (\(requested / 1_000_000)MB) exceeds system maximum (\(maxWirable / 1_000_000)MB). Clamping to maximum.")
            return maxWirable
        }

        if requested <= 0 {
            print("[WiredMemoryManager] WARNING: Requested wired limit is <= 0. Disabling wired memory.")
            return 0
        }

        return requested
    }
}
