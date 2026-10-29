import Foundation
import MLX
import MLXAudioCore
import Testing

@Suite("Mamba-2")
struct Mamba2Tests {
    @Test func groupedScanHasAnalyticState() {
        for device in [Device.cpu, Device.gpu] {
            Device.withDefaultDevice(device) {
                let x = MLXArray.ones([1, 2, 4, 1])
                let b = MLXArray([Float(0.5), 0.25, 0.5, 0.25]).reshaped([1, 2, 2, 1])
                let c = MLXArray([Float(0.25), 0.5, 0.25, 0.5]).reshaped([1, 2, 2, 1])
                let zero = MLXArray.zeros([4])
                let result = Mamba2Scan.callAsFunction(x: x, aLog: zero, b: b, c: c, d: zero,
                    dt: .zeros([1, 2, 4]), dtBias: zero)
                // A=-1, dt=log(2), decay=1/2. Heads 0/1 and 2/3 use
                // different B/C groups, with output dt/8 then 3*dt/16.
                let delta = log(Float(2))
                let expected = MLXArray(Array(repeating: delta / 8, count: 4)
                    + Array(repeating: 3 * delta / 16, count: 4)).reshaped(x.shape)
                let state = MLXArray([3 * delta / 4, 3 * delta / 4, 3 * delta / 8, 3 * delta / 8])
                    .reshaped([1, 4, 1, 1])
                #expect(max(abs(result.output - expected)).item(Float.self) < 2e-8)
                #expect(max(abs(result.state - state)).item(Float.self) < 1e-7)
            }
        }
    }

    @Test func mambaBF16ScanMatchesCPUAndContinuesState() {
        func scan(on device: Device) -> (MLXArray, MLXArray) {
            Device.withDefaultDevice(device) {
                let x = (MLX.arange(12, dtype: .float32) / 16).reshaped([1, 3, 2, 2]).asType(.bfloat16)
                let b = MLXArray.ones([1, 3, 1, 32], dtype: .bfloat16) * 0.125
                let c = b * 0.5
                let a = MLXArray.zeros([2], dtype: .bfloat16)
                let d = MLXArray.ones([2], dtype: .bfloat16) * 0.25
                let dt = MLXArray.zeros([1, 3, 2], dtype: .bfloat16)
                let full = Mamba2Scan.callAsFunction(x: x, aLog: a, b: b, c: c, d: d, dt: dt, dtBias: a)
                let first = Mamba2Scan.callAsFunction(x: x[0..., 0..<2, 0..., 0...], aLog: a,
                    b: b[0..., 0..<2, 0..., 0...], c: c[0..., 0..<2, 0..., 0...], d: d,
                    dt: dt[0..., 0..<2, 0...], dtBias: a)
                let last = Mamba2Scan.callAsFunction(x: x[0..., 2..<3, 0..., 0...], aLog: a,
                    b: b[0..., 2..<3, 0..., 0...], c: c[0..., 2..<3, 0..., 0...], d: d,
                    dt: dt[0..., 2..<3, 0...], dtBias: a, state: first.state)
                #expect(max(abs(full.state - last.state)).item(Float.self) < 1e-7)
                #expect(full.output.dtype == .bfloat16 && last.state.dtype == .float32)
                eval(full.output, full.state)
                return (full.output, full.state)
            }
        }
        let cpu = scan(on: .cpu), gpu = scan(on: .gpu)
        #expect(max(abs(cpu.0 - gpu.0)).item(Float.self) == 0)
        #expect(max(abs(cpu.1 - gpu.1)).item(Float.self) < 1e-7)
    }

}
