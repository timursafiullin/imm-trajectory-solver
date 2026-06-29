package dev.trajectory.imm.measurement

import dev.trajectory.imm.domain.CovarianceMatrix
import dev.trajectory.imm.domain.CartesianTimedMeasurement
import dev.trajectory.imm.domain.MeasurementEstimate
import dev.trajectory.imm.domain.StateVector
import dev.trajectory.imm.math.Matrix

sealed interface MeasurementModel {
    val measurementDimension: Int

    fun predictMeasurement(
        state: StateVector,
        time: Double,
    ): MeasurementEstimate

    fun measurementNoise(): CovarianceMatrix

    fun measurementNoise(measurement: CartesianTimedMeasurement): CovarianceMatrix = measurementNoise()
}

interface LinearMeasurementModel : MeasurementModel {
    fun measurementMatrix(): Matrix
}

interface EkfMeasurementModel : MeasurementModel {
    fun jacobian(state: StateVector): Matrix
}

interface UnscentedMeasurementModel : MeasurementModel
