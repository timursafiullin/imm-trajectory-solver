package dev.trajectory.imm.filter

import dev.trajectory.imm.domain.CovarianceMatrix
import dev.trajectory.imm.domain.MeasurementEstimate
import dev.trajectory.imm.domain.MeasurementVector
import dev.trajectory.imm.domain.StateEstimate
import dev.trajectory.imm.domain.TimedMeasurement

data class FilterState(
    val estimate: StateEstimate,
)

interface TrackingFilter {
    val name: String
    val stateDimension: Int
    val measurementDimension: Int

    fun initialize(measurements: List<TimedMeasurement>): FilterState

    fun predict(
        state: FilterState,
        toTime: Double,
    ): FilterPrediction

    fun correct(
        prediction: FilterPrediction,
        measurement: TimedMeasurement,
        gatingThreshold: Double,
    ): FilterUpdate

    fun expectedMeasurement(estimate: StateEstimate): MeasurementEstimate
}

data class FilterPrediction(
    val previousState: FilterState,
    val predictedEstimate: StateEstimate,
    val predictedMeasurement: MeasurementEstimate,
    val predictedMeasurementCovariance: CovarianceMatrix,
)

data class FilterUpdate(
    val prediction: FilterPrediction,
    val updatedEstimate: StateEstimate,
    val innovation: MeasurementVector,
    val innovationCovariance: CovarianceMatrix,
    val logLikelihood: Double,
    val mahalanobisDistanceSquared: Double,
    val gatingThreshold: Double,
    val accepted: Boolean,
)
