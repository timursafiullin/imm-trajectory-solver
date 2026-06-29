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

data class Wgs84Position(
    val latitudeDegrees: Double,
    val longitudeDegrees: Double,
    val heightMeters: Double,
) {
    init {
        require(latitudeDegrees.isFinite()) { "WGS84 latitude must be finite." }
        require(longitudeDegrees.isFinite()) { "WGS84 longitude must be finite." }
        require(heightMeters.isFinite()) { "WGS84 height must be finite." }
        require(latitudeDegrees in -90.0..90.0) {
            "WGS84 latitude must be in [-90, 90] degrees, got $latitudeDegrees."
        }
        require(longitudeDegrees in -180.0..180.0) {
            "WGS84 longitude must be in [-180, 180] degrees, got $longitudeDegrees."
        }
    }
}

sealed interface TimedMeasurement<out P> {
    val time: Double
    val position: P
}

data class CartesianTimedMeasurement(
    override val time: Double,
    override val position: Vector3,
    val measurementNoiseOverride: CovarianceMatrix? = null,
) : TimedMeasurement<Vector3> {
    init {
        require(time.isFinite()) { "Measurement time must be finite." }
    }
}

data class Wgs84TimedMeasurement(
    override val time: Double,
    override val position: Wgs84Position,
) : TimedMeasurement<Wgs84Position> {
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

sealed interface PositionCovariance

data class CartesianPositionCovariance(
    val covariance: CovarianceMatrix,
) : PositionCovariance

data class Wgs84PositionCovariance(
    val enuCovariance: CovarianceMatrix,
    val geodeticCovariance: CovarianceMatrix,
) : PositionCovariance

sealed interface PositionMeasurementNoise

data class EnuPositionNoise(
    val covariance: CovarianceMatrix,
) : PositionMeasurementNoise {
    companion object {
        fun isotropic(variance: Double): EnuPositionNoise {
            require(variance.isFinite() && variance > 0.0) {
                "ENU measurement variance must be finite and positive."
            }
            return EnuPositionNoise(
                CovarianceMatrix(Matrix.diagonal(listOf(variance, variance, variance))),
            )
        }
    }
}

data class Wgs84PositionNoise(
    val covariance: CovarianceMatrix,
) : PositionMeasurementNoise {
    init {
        require(covariance.value.rows == 3) { "WGS84 position noise covariance must be 3x3." }
    }
}

data class TrajectoryPrediction<P>(
    val time: Double,
    val expectedPosition: P,
    val covariance: PositionCovariance,
    val internalStateEstimate: StateEstimate,
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

data class TrajectoryUpdate<P>(
    val time: Double,
    val prediction: TrajectoryPrediction<P>,
    val correctedPosition: P,
    val internalUpdatedEstimate: StateEstimate,
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
