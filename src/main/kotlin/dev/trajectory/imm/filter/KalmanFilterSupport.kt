package dev.trajectory.imm.filter

import dev.trajectory.imm.domain.CovarianceMatrix
import dev.trajectory.imm.domain.MeasurementEstimate
import dev.trajectory.imm.domain.MeasurementVector
import dev.trajectory.imm.domain.StateEstimate
import dev.trajectory.imm.domain.StateVector
import dev.trajectory.imm.domain.TimedMeasurement
import dev.trajectory.imm.math.Matrix
import dev.trajectory.imm.measurement.LinearMeasurementModel
import dev.trajectory.imm.state.CanonicalKinematicState
import kotlin.math.ln

internal data class KalmanInnovationStats(
    val logLikelihood: Double,
    val mahalanobisDistanceSquared: Double,
)

internal fun initializeCanonicalKinematicState(
    filterName: String,
    measurements: List<TimedMeasurement>,
    initializationConfig: LinearKalmanInitializationConfig,
): FilterState {
    require(measurements.isNotEmpty()) { "Filter $filterName initialization requires at least one measurement." }
    require(measurements.zipWithNext().all { (left, right) -> right.time > left.time }) {
        "Filter $filterName initialization measurements must be strictly increasing in time."
    }
    val last = measurements.last()
    val state = DoubleArray(CanonicalKinematicState.DIMENSION)
    state[CanonicalKinematicState.PX] = last.position.x
    state[CanonicalKinematicState.PY] = last.position.y
    state[CanonicalKinematicState.PZ] = last.position.z

    if (measurements.size >= 2) {
        val previous = measurements[measurements.lastIndex - 1]
        val dt = last.time - previous.time
        state[CanonicalKinematicState.VX] = (last.position.x - previous.position.x) / dt
        state[CanonicalKinematicState.VY] = (last.position.y - previous.position.y) / dt
        state[CanonicalKinematicState.VZ] = (last.position.z - previous.position.z) / dt
    }

    if (measurements.size >= 3) {
        val first = measurements[measurements.lastIndex - 2]
        val middle = measurements[measurements.lastIndex - 1]
        val dt1 = middle.time - first.time
        val dt2 = last.time - middle.time
        val vx1 = (middle.position.x - first.position.x) / dt1
        val vy1 = (middle.position.y - first.position.y) / dt1
        val vz1 = (middle.position.z - first.position.z) / dt1
        val vx2 = (last.position.x - middle.position.x) / dt2
        val vy2 = (last.position.y - middle.position.y) / dt2
        val vz2 = (last.position.z - middle.position.z) / dt2
        val accelerationDt = (dt1 + dt2) * 0.5
        state[CanonicalKinematicState.AX] = (vx2 - vx1) / accelerationDt
        state[CanonicalKinematicState.AY] = (vy2 - vy1) / accelerationDt
        state[CanonicalKinematicState.AZ] = (vz2 - vz1) / accelerationDt
    }

    val covariance = Matrix.diagonal(
        listOf(
            initializationConfig.initialPositionVariance,
            initializationConfig.initialPositionVariance,
            initializationConfig.initialPositionVariance,
            initializationConfig.initialVelocityVariance,
            initializationConfig.initialVelocityVariance,
            initializationConfig.initialVelocityVariance,
            initializationConfig.initialAccelerationVariance,
            initializationConfig.initialAccelerationVariance,
            initializationConfig.initialAccelerationVariance,
        ),
    )
    return FilterState(
        StateEstimate(
            time = last.time,
            mean = StateVector(Matrix.columnVector(state.toList())),
            covariance = CovarianceMatrix(covariance),
        ),
    )
}

internal fun requireCanonicalEstimate(
    filterName: String,
    estimate: StateEstimate,
) {
    require(estimate.mean.value.rows == CanonicalKinematicState.DIMENSION) {
        "Filter $filterName expected canonical ${CanonicalKinematicState.DIMENSION}D state."
    }
    require(estimate.covariance.value.rows == CanonicalKinematicState.DIMENSION) {
        "Filter $filterName expected canonical ${CanonicalKinematicState.DIMENSION}D covariance."
    }
}

internal fun measurementVector(measurement: TimedMeasurement): Matrix {
    return Matrix.columnVector(listOf(measurement.position.x, measurement.position.y, measurement.position.z))
}

internal fun expectedLinearMeasurement(
    measurementModel: LinearMeasurementModel,
    estimate: StateEstimate,
): MeasurementEstimate {
    val h = measurementModel.measurementMatrix()
    val covariance = (h * estimate.covariance.value * h.transpose() + measurementModel.measurementNoise().value).symmetrized()
    return MeasurementEstimate(
        time = estimate.time,
        mean = MeasurementVector(h * estimate.mean.value),
        covariance = CovarianceMatrix(covariance),
    )
}

internal fun gaussianInnovationStats(
    measurementDimension: Int,
    innovation: Matrix,
    innovationCovariance: Matrix,
): KalmanInnovationStats {
    val solvedInnovation = innovationCovariance.solve(innovation)
    val mahalanobis = (innovation.transpose() * solvedInnovation)[0, 0]
    val logLikelihood = -0.5 * (
        measurementDimension * ln(2.0 * Math.PI) +
            innovationCovariance.logDeterminantPositive() +
            mahalanobis
        )
    return KalmanInnovationStats(
        logLikelihood = logLikelihood,
        mahalanobisDistanceSquared = mahalanobis,
    )
}

internal fun josephUpdatedEstimate(
    time: Double,
    predictedMean: Matrix,
    predictedCovariance: Matrix,
    measurementMatrix: Matrix,
    measurementNoise: Matrix,
    innovation: Matrix,
    stateDimension: Int,
): StateEstimate {
    val innovationCovariance = (measurementMatrix * predictedCovariance * measurementMatrix.transpose() + measurementNoise).symmetrized()
    val kalmanGain = predictedCovariance * measurementMatrix.transpose() * innovationCovariance.inverse()
    val updatedMean = predictedMean + kalmanGain * innovation
    val identity = Matrix.identity(stateDimension)
    val residual = identity - kalmanGain * measurementMatrix
    val updatedCovariance = (
        residual * predictedCovariance * residual.transpose() +
            kalmanGain * measurementNoise * kalmanGain.transpose()
        ).symmetrized()
    return StateEstimate(
        time = time,
        mean = StateVector(updatedMean),
        covariance = CovarianceMatrix(updatedCovariance),
    )
}
