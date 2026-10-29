// Copyright © 2025 Apple Inc.
// Adapted from mlx-lm's Mamba-2 state update under the MIT License.

import MLX
import MLXFast
import MLXNN

/// Mamba-2 selective state scan. Inputs use [batch, time, head, channel], with
/// grouped B/C [batch, time, group, state]. The returned recurrent state is FP32
/// [batch, head, channel, state], independent of activation precision.
public enum Mamba2Scan {
    private static let kernel = MLXFast.metalKernel(
        name: "mamba2_state_scan",
        inputNames: ["x", "decay", "B", "C", "D", "dt", "state_in"],
        outputNames: ["y", "state_out"],
        source: """
            auto n = thread_position_in_grid.z;
            auto batch = n / H;
            auto head = n % H;
            auto group = head / (H / G);
            auto d = thread_position_in_grid.y;
            auto lane = thread_position_in_threadgroup.x;
            constexpr int N = (S + 31) / 32;
            float state[N];
            for (int i = 0; i < N; ++i) {
                auto s = N * lane + i;
                state[i] = s < S ? state_in[(n * P + d) * S + s] : 0.0f;
            }
            for (int t = 0; t < L; ++t) {
                auto th = (batch * L + t) * H + head;
                auto tg = (batch * L + t) * G + group;
                auto input = static_cast<float>(x[th * P + d]);
                auto scale = input * dt[th];
                float sum = 0.0f;
                for (int i = 0; i < N; ++i) {
                    auto s = N * lane + i;
                    if (s < S) {
                        state[i] = decay[th] * state[i] + scale * static_cast<float>(B[tg * S + s]);
                        sum += state[i] * static_cast<float>(C[tg * S + s]);
                    }
                }
                sum = simd_sum(sum);
                if (thread_index_in_simdgroup == 0) y[th * P + d] = static_cast<T>(sum + input * static_cast<float>(D[head]));
            }
            for (int i = 0; i < N; ++i) {
                auto s = N * lane + i;
                if (s < S) state_out[(n * P + d) * S + s] = state[i];
            }
            """)

    public static func callAsFunction(
        x: MLXArray, aLog: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray,
        dt: MLXArray, dtBias: MLXArray, state: MLXArray? = nil
    ) -> (output: MLXArray, state: MLXArray) {
        let batch = x.dim(0), length = x.dim(1), heads = x.dim(2), channels = x.dim(3)
        let groups = b.dim(2), stateSize = b.dim(3)
        let delta = softplus(dt.asType(.float32) + dtBias)
        // Match mlx-lm: initial prefill computes A in its stored precision;
        // recurrent single-token updates compute the exponent in Float32.
        let a = state != nil && length == 1 ? exp(aLog.asType(.float32)) : exp(aLog)
        let decay = exp(-a * delta)
        var memory = state ?? MLXArray.zeros([batch, heads, channels, stateSize], dtype: .float32)
        if Device.defaultDevice() == .gpu {
            let out = kernel([x, decay, b, c, d, delta, memory],
                template: [("T", x.dtype), ("L", length), ("H", heads), ("P", channels), ("G", groups), ("S", stateSize)],
                grid: (32, channels, batch * heads), threadGroup: (32, 4, 1),
                outputShapes: [x.shape, memory.shape], outputDTypes: [x.dtype, .float32])
            return (out[0], out[1])
        }
        let indices = floorDivide(MLX.arange(heads), heads / groups)
        var output: [MLXArray] = []
        for t in 0..<length {
            let input = x[0..., t, 0..., 0...].asType(.float32)
            let bt = take(b[0..., t, 0..., 0...], indices, axis: 1).asType(.float32)
            let ct = take(c[0..., t, 0..., 0...], indices, axis: 1).asType(.float32)
            memory = memory * decay[0..., t, 0...].expandedDimensions(axes: [2, 3])
                + (input * delta[0..., t, 0...].expandedDimensions(axis: -1)).expandedDimensions(axis: -1)
                * bt.expandedDimensions(axis: 2)
            output.append(((memory * ct.expandedDimensions(axis: 2)).sum(axis: -1)
                + input * d.expandedDimensions(axes: [0, 2])).asType(x.dtype))
        }
        return (stacked(output, axis: 1), memory)
    }
}
