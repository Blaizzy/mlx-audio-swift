func moduloFloatFixtureValues(
    count: Int,
    multiplier: Int = 1,
    modulus: Int,
    subtracting offset: Int = 0,
    divisor: Float
) -> [Float] {
    var values: [Float] = []
    values.reserveCapacity(count)
    for index in 0..<count {
        let remainder = (index * multiplier) % modulus
        values.append(Float(remainder - offset) / divisor)
    }
    return values
}
