package dev.trajectory.imm.domain

import dev.trajectory.imm.math.Matrix

@JvmInline
value class StateVector(
    val value: Matrix,
) {
    init {
        require(value.columns == 1) { "State vector must be a column vector, got ${value.rows}x${value.columns}." }
    }
}

@JvmInline
value class MeasurementVector(
    val value: Matrix,
) {
    init {
        require(value.columns == 1) { "Measurement vector must be a column vector, got ${value.rows}x${value.columns}." }
    }
}

@JvmInline
value class CovarianceMatrix(
    val value: Matrix,
) {
    init {
        require(value.rows == value.columns) {
            "Covariance matrix must be square, got ${value.rows}x${value.columns}."
        }
    }
}

data class Vector3(
    val x: Double,
    val y: Double,
    val z: Double,
) {
    init {
        require(x.isFinite() && y.isFinite() && z.isFinite()) { "Vector3 components must be finite." }
    }
}

data class TimedMeasurement(
    val time: Double,
    val position: Vector3,
) {
    init {
        require(time.isFinite()) { "Measurement time must be finite." }
    }
}

data class MeasurementEstimate(
    val time: Double,
    val mean: MeasurementVector,
    val covariance: CovarianceMatrix,
) {
    init {
        require(time.isFinite()) { "Measurement estimate time must be finite." }
        require(mean.value.rows == covariance.value.rows) {
            "Measurement mean dimension ${mean.value.rows} must match covariance size ${covariance.value.rows}."
        }
    }
}

data class StateEstimate(
    val time: Double,
    val mean: StateVector,
    val covariance: CovarianceMatrix,
) {
    init {
        require(time.isFinite()) { "State estimate time must be finite." }
        require(mean.value.rows == covariance.value.rows) {
            "State mean dimension ${mean.value.rows} must match covariance size ${covariance.value.rows}."
        }
    }
}

data class Prediction(
    val time: Double,
    val expectedMeasurement: MeasurementEstimate,
    val stateEstimate: StateEstimate,
    val modelProbabilities: Map<String, Double>,
    val metadata: Map<String, Double> = emptyMap(),
) {
    init {
        require(time.isFinite()) { "Prediction time must be finite." }
        require(modelProbabilities.isNotEmpty()) { "Prediction model probabilities must not be empty." }
        require(modelProbabilities.values.all { it.isFinite() }) { "Prediction model probabilities must be finite." }
        require(metadata.values.all { it.isFinite() }) { "Prediction metadata values must be finite." }
    }
}

data class UpdateResult(
    val time: Double,
    val prediction: Prediction,
    val updatedEstimate: StateEstimate,
    val accepted: Boolean,
    val logLikelihood: Double,
    val mahalanobisDistanceSquared: Double,
    val modelProbabilities: Map<String, Double>,
    val metadata: Map<String, Double> = emptyMap(),
) {
    init {
        require(time.isFinite()) { "Update result time must be finite." }
        require(logLikelihood.isFinite()) { "Update log-likelihood must be finite." }
        require(mahalanobisDistanceSquared.isFinite() && mahalanobisDistanceSquared >= 0.0) {
            "Update Mahalanobis distance squared must be finite and non-negative."
        }
        require(modelProbabilities.isNotEmpty()) { "Update model probabilities must not be empty." }
        require(modelProbabilities.values.all { it.isFinite() }) { "Update model probabilities must be finite." }
        require(metadata.values.all { it.isFinite() }) { "Update metadata values must be finite." }
    }
}
