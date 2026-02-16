//
//  StreamingEncoder.swift
//  MLXAudioSTT
//
//  Created by Prince Canuma on 07/02/2026.
//

import Foundation
import MLX

/// Wraps `Qwen3ASRAudioEncoder` for incremental encoding.
///
/// Accumulates mel frames until a full window (800 frames = ~8s audio) is ready,
/// then encodes via `encoder.encodeSingleWindow()`. Each encoded window produces
/// ~104 encoder tokens of dimension 2048.
/// Consecutive windows can overlap by a configurable number of mel frames.
///
/// Since the encoder uses block attention (no cross-window attention), new windows
/// can be encoded independently and concatenated with previous results.
public class StreamingEncoder {
    private let encoder: Qwen3ASRAudioEncoder
    private let windowSize: Int  // nWindowInfer = 800 mel frames
    private let windowStride: Int
    private let maxCachedWindows: Int

    /// Cached encoded window outputs
    private var cachedWindows: [MLXArray] = []

    /// Newly encoded full windows not yet consumed by the session.
    private var newlyEncodedWindows: [MLXArray] = []

    /// Total number of *completed* windows encoded since reset.
    ///
    /// This is monotonic and does not decrease when old cached windows are dropped.
    private var totalEncodedWindows: Int = 0

    /// Pending mel frames not yet forming a full window
    private var pendingFrames: MLXArray?

    /// Number of pending mel frames
    private var pendingFrameCount: Int = 0

    public init(
        encoder: Qwen3ASRAudioEncoder,
        maxCachedWindows: Int = 60,
        overlapFrames: Int = 0
    ) {
        self.encoder = encoder
        self.windowSize = encoder.nWindowInfer  // 800
        let clampedOverlap = max(0, min(overlapFrames, max(0, encoder.nWindowInfer - 1)))
        self.windowStride = max(1, encoder.nWindowInfer - clampedOverlap)
        self.maxCachedWindows = maxCachedWindows
    }

    /// Feed mel frames to the encoder. Full windows are encoded immediately.
    ///
    /// - Parameter melFrames: New mel frames `[numFrames, nMels]`
    /// - Returns: Number of new windows encoded
    public func feed(melFrames: MLXArray) -> Int {
        // Append to pending
        if let existing = pendingFrames {
            pendingFrames = MLX.concatenated([existing, melFrames], axis: 0)
        } else {
            pendingFrames = melFrames
        }
        pendingFrameCount = pendingFrames?.dim(0) ?? 0

        // Encode complete windows
        var newWindows = 0
        while pendingFrameCount >= windowSize {
            guard let frames = pendingFrames else { break }

            let windowFrames = frames[0..<windowSize]
            let encoded = encoder.encodeSingleWindow(windowFrames)
            eval(encoded)

            cachedWindows.append(encoded)
            newlyEncodedWindows.append(encoded)
            totalEncodedWindows += 1
            newWindows += 1

            // Trim pending
            if pendingFrameCount > windowStride {
                pendingFrames = frames[windowStride...]
                pendingFrameCount = pendingFrames!.dim(0)
            } else {
                pendingFrames = nil
                pendingFrameCount = 0
            }

            // Enforce max cache size
            if cachedWindows.count > maxCachedWindows {
                cachedWindows.removeFirst()
            }
        }

        return newWindows
    }

    /// Encode remaining partial window at session end.
    /// - Returns: Number of new windows encoded (0 or 1)
    public func flushPartial() -> Int {
        guard let frames = pendingFrames, pendingFrameCount > 0 else { return 0 }

        let encoded = encoder.encodeSingleWindow(frames)
        eval(encoded)
        cachedWindows.append(encoded)

        pendingFrames = nil
        pendingFrameCount = 0

        if cachedWindows.count > maxCachedWindows {
            cachedWindows.removeFirst()
        }

        return 1
    }

    /// Get concatenated encoder output from all cached windows.
    /// - Returns: `[totalTokens, outputDim]` or nil if no windows cached
    public func getCachedEncoderOutput() -> MLXArray? {
        guard !cachedWindows.isEmpty else { return nil }
        if cachedWindows.count == 1 {
            return cachedWindows[0]
        }
        return MLX.concatenated(cachedWindows, axis: 0)
    }

    /// Get concatenated encoder output starting from a specific window index.
    /// - Parameter startWindow: Index of the first cached window to include
    /// - Returns: `[totalTokens, outputDim]` or nil
    public func getCachedEncoderOutput(fromWindow startWindow: Int) -> MLXArray? {
        let start = max(0, startWindow)
        guard start < cachedWindows.count else { return nil }
        let slice = Array(cachedWindows[start...])
        if slice.count == 1 { return slice[0] }
        return MLX.concatenated(slice, axis: 0)
    }

    /// Encode the current pending partial window for early feedback.
    ///
    /// This does NOT consume the pending frames — they remain in the buffer
    /// and will be re-encoded as part of the full window when it completes.
    /// Cost: ~50ms per call, but only for the latest incomplete window.
    ///
    /// - Returns: Encoded partial window `[tokens, outputDim]` or nil
    public func encodePending() -> MLXArray? {
        guard let frames = pendingFrames, pendingFrameCount > 0 else { return nil }

        let encoded = encoder.encodeSingleWindow(frames)
        eval(encoded)
        return encoded
    }

    /// Get full encoder output including pending partial window.
    /// - Parameter fromWindow: Optional start window index for windowed decode
    /// - Returns: `[totalTokens, outputDim]` or nil
    public func getFullEncoderOutput(fromWindow: Int? = nil) -> MLXArray? {
        let cached: MLXArray?
        if let fromWindow {
            cached = getCachedEncoderOutput(fromWindow: fromWindow)
        } else {
            cached = getCachedEncoderOutput()
        }
        let pending = encodePending()

        switch (cached, pending) {
        case (nil, nil):
            return nil
        case (let c?, nil):
            return c
        case (nil, let p?):
            return p
        case (let c?, let p?):
            return MLX.concatenated([c, p], axis: 0)
        }
    }

    /// Number of fully encoded windows.
    public var encodedWindowCount: Int {
        totalEncodedWindows
    }

    /// Whether there are pending frames not yet encoded as a full window.
    public var hasPendingFrames: Bool {
        pendingFrameCount > 0
    }

    /// Drain and return newly encoded full windows since the last drain.
    public func drainNewlyEncodedWindows() -> [MLXArray] {
        guard !newlyEncodedWindows.isEmpty else { return [] }
        defer { newlyEncodedWindows.removeAll(keepingCapacity: true) }
        return newlyEncodedWindows
    }

    /// Total encoder tokens across all cached windows.
    public var totalCachedTokens: Int {
        cachedWindows.reduce(0) { $0 + $1.dim(0) }
    }

    /// Reset all state for a new session.
    public func reset() {
        cachedWindows.removeAll()
        newlyEncodedWindows.removeAll()
        totalEncodedWindows = 0
        pendingFrames = nil
        pendingFrameCount = 0
    }
}
