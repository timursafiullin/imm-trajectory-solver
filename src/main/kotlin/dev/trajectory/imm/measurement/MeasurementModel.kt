package dev.trajectory.imm.measurement

import dev.trajectory.imm.domain.CovarianceMatrix
import dev.trajectory.imm.domain.MeasurementEstimate
import dev.trajectory.imm.domain.MeasurementVector
import dev.trajectory.imm.domain.StateVector
import dev.trajectory.imm.math.Matrix
import dev.trajectory.imm.state.CanonicalKinematicState

sealed interface MeasurementModel {
    val measurementDimension: Int

    fun predictMeasurement(
        state: StateVector,
        time: Double,
    ): MeasurementEstimate

    fun measurementNoise(): CovarianceMatrix
}

interface LinearMeasurementModel : MeasurementModel {
    fun measurementMatrix(): Matrix
}

interface EkfMeasurementModel : MeasurementModel {
    fun jacobian(state: StateVector): Matrix
}

interface UnscentedMeasurementModel : MeasurementModel

class CartesianPositionMeasurementModel(
    private val noiseCovariance: CovarianceMatrix,
) : LinearMeasurementModel {
    init {
        require(noiseCovariance.value.rows == MEASUREMENT_DIMENSION) {
            "Cartesian measurement covariance must be ${MEASUREMENT_DIMENSION}x$MEASUREMENT_DIMENSION."
        }
    }

    override val measurementDimension: Int = MEASUREMENT_DIMENSION

    override fun predictMeasurement(state: StateVector, time: Double): MeasurementEstimate {
        require(time.isFinite()) { "Measurement prediction time must be finite." }
        require(state.value.rows == CanonicalKinematicState.DIMENSION) {
            "Cartesian measurement model expects canonical ${CanonicalKinematicState.DIMENSION}D state."
        }
        val mean = measurementMatrix() * state.value
        return MeasurementEstimate(
            time = time,
            mean = MeasurementVector(mean),
            covariance = noiseCovariance,
        )
    }

    override fun measurementNoise(): CovarianceMatrix = noiseCovariance

    override fun measurementMatrix(): Matrix {
        return Matrix.ofRows(
            listOf(
                listOf(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                listOf(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                listOf(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ),
        )
    }

    companion object {
        const val MEASUREMENT_DIMENSION: Int = 3

        fun isotropic(variance: Double): CartesianPositionMeasurementModel {
            require(variance.isFinite() && variance > 0.0) { "Measurement variance must be finite and positive." }
            return CartesianPositionMeasurementModel(
                CovarianceMatrix(Matrix.diagonal(listOf(variance, variance, variance))),
            )
        }
    }
}
