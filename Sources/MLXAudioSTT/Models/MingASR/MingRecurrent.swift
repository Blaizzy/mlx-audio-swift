// Copyright © 2025 Apple Inc.
// Adapted from mlx-lm under the MIT License.

import MLX
import MLXFast

// Vector-gated delta recurrence adapted from mlx-lm's gated_delta.py (MIT).
enum MingDeltaKernel {
    private static let kernel = MLXFast.metalKernel(
        name: "ming_gated_delta", inputNames: ["q", "k", "v", "g", "beta", "state_in"],
        outputNames: ["y", "state_out"],
        source: """
            auto n = thread_position_in_grid.z;
            auto b = n / H;
            auto h = n % H;
            constexpr int N = D / 32;
            auto q_ = q + (b * T * H + h) * D;
            auto k_ = k + (b * T * H + h) * D;
            auto v_ = v + (b * T * H + h) * D;
            auto g_ = g + (b * T * H + h) * D;
            auto beta_ = beta + b * T * H;
            y += (b * T * H + h) * D;
            auto lane = thread_position_in_threadgroup.x;
            auto dv = thread_position_in_grid.y;
            auto si = state_in + (n * D + dv) * D;
            auto so = state_out + (n * D + dv) * D;
            float s[N];
            for (int i = 0; i < N; ++i) s[i] = si[N * lane + i];
            for (int t = 0; t < T; ++t) {
                float memory = 0.0f;
                for (int i = 0; i < N; ++i) {
                    auto j = N * lane + i;
                    s[i] = s[i] * g_[j];
                    memory += s[i] * k_[j];
                }
                memory = simd_sum(memory);
                auto delta = (float(v_[dv]) - memory) * beta_[h];
                float out = 0.0f;
                for (int i = 0; i < N; ++i) {
                    auto j = N * lane + i;
                    s[i] = s[i] + k_[j] * delta;
                    out += s[i] * q_[j];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) y[dv] = out;
                q_ += H * D; k_ += H * D; v_ += H * D; g_ += H * D;
                y += H * D; beta_ += H;
            }
            for (int i = 0; i < N; ++i) so[N * lane + i] = s[i];
            """)

    static func callAsFunction(
        q: MLXArray, k: MLXArray, v: MLXArray, decay: MLXArray, beta: MLXArray,
        state: MLXArray
    ) -> (MLXArray, MLXArray) {
        let b = q.dim(0)
        let t = q.dim(1)
        let h = q.dim(2)
        let d = q.dim(3)
        guard d % 32 == 0, Device.defaultDevice() == .gpu else {
            return reference(q: q, k: k, v: v, decay: decay, beta: beta, state: state)
        }
        let out = kernel(
            [q, k, v, decay, beta, state], template: [("T", t), ("H", h), ("D", d)],
            grid: (32, d, b * h), threadGroup: (32, 4, 1),
            outputShapes: [v.shape, state.shape], outputDTypes: [.float32, .float32])
        return (out[0], out[1])
    }

    static func reference(
        q: MLXArray, k: MLXArray, v: MLXArray, decay: MLXArray, beta: MLXArray,
        state initial: MLXArray
    ) -> (MLXArray, MLXArray) {
        var state = initial
        var output: [MLXArray] = []
        for i in 0 ..< q.dim(1) {
            let key = k[0..., i, 0..., 0...].expandedDimensions(axis: -2)
            state = state * decay[0..., i, 0..., 0...].expandedDimensions(axis: -2)
            let memory = (state * key).sum(axis: -1)
            let delta =
                (v[0..., i, 0..., 0...].asType(.float32) - memory)
                * beta[0..., i, 0...].expandedDimensions(axis: -1)
            state = state + key * delta.expandedDimensions(axis: -1)
            output.append(
                (state * q[0..., i, 0..., 0...].expandedDimensions(axis: -2)).sum(axis: -1))
        }
        return (stacked(output, axis: 1), state)
    }
}
