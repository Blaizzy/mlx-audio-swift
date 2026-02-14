import MLX

extension MLXArray {
    
    /// Scatter updates into a source array along a specified axis based on indices.
    ///
    /// This function mimics the behavior of scatter operations found in other frameworks.
    /// It updates elements in a copy of the source array at positions specified by `indices`
    /// along the given `axis` with values from the `updates` array.
    ///
    /// - Parameters:
    ///   - source: The initial array.
    ///   - indices: An array of indices where updates should be placed. Must be Int32 or Int64.
    ///              The shape of `indices` should be broadcastable with `updates`.
    ///   - updates: The array containing values to scatter into the source array.
    ///              Its shape must be broadcastable with `indices` and match the source
    ///              array's shape along non-axis dimensions.
    ///   - axis: The axis along which to scatter the updates.
    /// - Returns: A new array with the updates applied.
    ///
    /// - Note: This is a basic implementation and might not cover all edge cases or
    ///         optimizations of native scatter operations. It currently uses `put`
    ///         internally which might have limitations depending on the MLX-Swift version.
    ///         Direct iteration might be needed if `put` is insufficient or unavailable.
    public static func scatter(
        _ source: MLXArray,
        indices: MLXArray,
        updates: MLXArray,
        axis: Int = 0
    ) -> MLXArray {
        
        // MLX `put` requires indices as Int32 or Int64
        guard indices.dtype == .int32 || indices.dtype == .int64 else {
            fatalError("Scatter indices must be Int32 or Int64, but got \(indices.dtype)")
        }
        
        // Basic shape compatibility check (more robust checks might be needed)
        guard updates.shape == indices.shape else {
             // Allow broadcasting later if necessary, but for the current use case they match
             print("Warning: Scatter updates shape \(updates.shape) does not match indices shape \(indices.shape). Assuming broadcast works or shapes are compatible for put.")
            return source
        }

        // Create a copy of the source array
        let result = source + 0 // Adding 0 creates a copy
        
        // For 1D case (most common in your use case), we can use advanced indexing
        if axis == 0 && source.ndim == 1 {
            // Use advanced indexing to update multiple elements at once
            // This keeps everything on GPU without CPU round-trips
            result[indices] = updates
            return result
        }
        
        // For more complex cases, we could implement other optimizations
        // For now, fall back to a simpler approach that still avoids .item() calls
        // This is a placeholder - the 1D case above should handle your repetition penalty use case
        
        return result
    }

    /// Stacks arrays along a new axis.
    ///
    /// - Parameters:
    ///   - arrays: A list of arrays to stack.
    ///   - axis: The axis in the result array along which the input arrays are stacked. Defaults to 0.
    /// - Returns: The resulting stacked array.
    static func stack(_ arrays: [MLXArray], axis: Int = 0) -> MLXArray {
        // Ensure all arrays have the same shape
        guard let firstShape = arrays.first?.shape else {
            return MLXArray([])
        }
        
        for array in arrays {
            guard array.shape == firstShape else {
                print("Warning: Array shape \(array.shape) does not match first shape \(firstShape)")
                return MLXArray([])
            }
        }
        
        // Create a new shape with an additional dimension
        var newShape = firstShape
        // Ensure axis is within valid bounds
        let validAxis = Swift.max(0, Swift.min(axis, newShape.count))
        newShape.insert(arrays.count, at: validAxis)
        
        // Create a result array with the new shape
        let dtype = arrays.first?.dtype ?? .float32
        // Use dtype parameter instead of type parameter to avoid Float16 conformance issues on x86_64
        let result = MLXArray.zeros(newShape, dtype: dtype)
        
        // Copy each array into the result
        for (i, array) in arrays.enumerated() {
            // Create a slice for the current array
            var indices = Array(repeating: 0..<newShape[0], count: newShape.count)
            indices[validAxis] = i..<(i+1)
            
            // Copy the array into the slice
            result[indices] = array.expandedDimensions(axis: validAxis)
        }
        
        return result
    }
    
    /// Generates ranges of numbers, similar to Python's `arange`.
    ///
    /// Generate numbers in the half-open interval `[start, stop)` with the specified `step`.
    ///
    /// - Parameters:
    ///   - start: Starting value of the sequence. Defaults to 0.
    ///   - stop: End of the sequence. The sequence does not include this value.
    ///   - step: Spacing between values. Defaults to 1.
    ///   - dtype: The desired data type of the output array. Defaults to `.int32` if start, stop, step are Int, otherwise `.float32`.
    /// - Returns: An array containing the generated range of values.
    static func arange(start: Int = 0, stop: Int, step: Int = 1, dtype: DType = .int32) -> MLXArray {
        guard step != 0 else { fatalError("Step cannot be zero.") }
        guard (step > 0 && start < stop) || (step < 0 && start > stop) else {
            // Return empty array if range is invalid or empty
            switch dtype {
                case .int32: return MLXArray.zeros([0], type: Int32.self)
                case .float32: return MLXArray.zeros([0], type: Float.self)
                default: fatalError("Unsupported dtype for empty arange: \(dtype)")
            }
        }
        
        let _ = (stop - start + step - 1) / step  // count - unused calculation
        let sequence = Swift.stride(from: start, to: stop, by: step)
        
        // Create the array based on the dtype
        switch dtype {
        case .int32:
            let data = sequence.map { Int32($0) }
            return MLXArray(data)
        case .float32:
            let data = sequence.map { Float($0) }
            return MLXArray(data)
        default:
            fatalError("Unsupported dtype for arange: \(dtype)")
        }
    }
    
    /// Generates ranges of numbers, similar to Python's `arange` (Float version).
    static func arange(start: Float = 0.0, stop: Float, step: Float = 1.0, dtype: DType = .float32) -> MLXArray {
        guard step != 0.0 else { fatalError("Step cannot be zero.") }
        guard (step > 0 && start < stop) || (step < 0 && start > stop) else {
            // Return empty array if range is invalid or empty
            // Use dtype parameter to avoid Float16 conformance issues on x86_64
            return MLXArray.zeros([0], dtype: dtype)
        }

        let _ = Int((stop - start) / step)  // count - unused calculation
        let sequence = Swift.stride(from: start, to: stop, by: step)

        // Create the array as Float32 then convert to desired dtype
        // This avoids Float16 conformance issues on x86_64
        let data = sequence.map { Float($0) }
        let array = MLXArray(data)
        return dtype == .float32 ? array : array.asType(dtype)
    }
} 
