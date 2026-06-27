package dev.trajectory.imm.validation

import dev.trajectory.imm.math.Matrix
import kotlin.math.abs

object Validators {
    fun requireCanonicalDimension(name: String, dimension: Int, canonicalDimension: Int) {
        require(dimension == canonicalDimension) {
            "$name must expose canonical $canonicalDimension-dimensional state, got $dimension."
        }
    }

    fun requireTransitionMatrix(matrix: Matrix, expectedSize: Int, tolerance: Double = 1.0e-9) {
        require(matrix.rows == expectedSize && matrix.columns == expectedSize) {
            "Transition matrix must be ${expectedSize}x$expectedSize, got ${matrix.rows}x${matrix.columns}."
        }
        for (row in 0 until matrix.rows) {
            var sum = 0.0
            for (column in 0 until matrix.columns) {
                val value = matrix[row, column]
                require(value >= -tolerance) {
                    "Transition matrix entry [$row,$column] must be non-negative, got $value."
                }
                sum += value
            }
            require(abs(sum - 1.0) <= tolerance) {
                "Transition matrix row $row must sum to 1.0 within $tolerance, got $sum."
            }
        }
    }

    fun requireProbabilities(
        probabilities: Map<String, Double>,
        expectedNames: List<String>,
        tolerance: Double = 1.0e-9,
    ) {
        require(probabilities.keys == expectedNames.toSet()) {
            "Probability names ${probabilities.keys} must match expected names ${expectedNames.toSet()}."
        }
        var sum = 0.0
        probabilities.forEach { (name, probability) ->
            require(probability.isFinite()) { "Probability for $name must be finite." }
            require(probability >= -tolerance) { "Probability for $name must be non-negative, got $probability." }
            sum += probability
        }
        require(abs(sum - 1.0) <= tolerance) {
            "Probabilities must sum to 1.0 within $tolerance, got $sum."
        }
    }

    fun requireCovariance(matrix: Matrix, name: String, tolerance: Double = 1.0e-7) {
        require(matrix.rows == matrix.columns) { "$name covariance must be square, got ${matrix.rows}x${matrix.columns}." }
        require(matrix.isSymmetric(tolerance)) { "$name covariance must be symmetric within $tolerance." }
        require(matrix.isPositiveSemiDefinite(tolerance)) { "$name covariance must be positive semi-definite within $tolerance." }
    }
}

fun normalizeProbabilities(values: Map<String, Double>): Map<String, Double> {
    require(values.isNotEmpty()) { "Cannot normalize an empty probability map." }
    require(values.values.all { it.isFinite() && it >= 0.0 }) {
        "Probability values must be finite and non-negative."
    }
    val sum = values.values.sum()
    require(sum > 0.0) { "Probability sum must be positive." }
    return values.mapValues { (_, value) -> value / sum }
}

fun logSumExp(values: List<Double>): Double {
    require(values.isNotEmpty()) { "logSumExp requires at least one value." }
    val max = values.max()
    if (max == Double.NEGATIVE_INFINITY) {
        return max
    }
    var sum = 0.0
    for (value in values) {
        sum += kotlin.math.exp(value - max)
    }
    return max + kotlin.math.ln(sum)
}
