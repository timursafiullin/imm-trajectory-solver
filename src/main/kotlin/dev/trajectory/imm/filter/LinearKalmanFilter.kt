package dev.trajectory.imm.filter

import dev.trajectory.imm.domain.CovarianceMatrix
import dev.trajectory.imm.domain.MeasurementEstimate
import dev.trajectory.imm.domain.MeasurementVector
import dev.trajectory.imm.domain.StateEstimate
import dev.trajectory.imm.domain.StateVector
import dev.trajectory.imm.domain.TimedMeasurement
import dev.trajectory.imm.math.Matrix
import dev.trajectory.imm.measurement.LinearMeasurementModel
import dev.trajectory.imm.motion.MotionModel
import dev.trajectory.imm.state.CanonicalKinematicState
import kotlin.math.ln

data class LinearKalmanInitializationConfig(
    val initialPositionVariance: Double = 100.0,
    val initialVelocityVariance: Double = 100.0,
    val initialAccelerationVariance: Double = 400.0,
) {
    init {
        require(initialPositionVariance > 0.0 && initialPositionVariance.isFinite()) {
            "Initial position variance must be finite and positive."
        }
        require(initialVelocityVariance > 0.0 && initialVelocityVariance.isFinite()) {
            "Initial velocity variance must be finite and positive."
        }
        require(initialAccelerationVariance > 0.0 && initialAccelerationVariance.isFinite()) {
            "Initial acceleration variance must be finite and positive."
        }
    }
}

class LinearKalmanFilter(
    override val name: String,
    private val motionModel: MotionModel,
    private val measurementModel: LinearMeasurementModel,
    private val initializationConfig: LinearKalmanInitializationConfig = LinearKalmanInitializationConfig(),
) : TrackingFilter {
    init {
        require(name.isNotBlank()) { "Filter name must not be blank." }
        require(motionModel.stateDimension == CanonicalKinematicState.DIMENSION) {
            "MVP linear Kalman filters must expose canonical ${CanonicalKinematicState.DIMENSION}D state."
        }
        require(measurementModel.measurementDimension == 3) {
            "MVP linear Kalman filter expects 3D Cartesian measurements."
        }
    }

    override val stateDimension: Int = motionModel.stateDimension
    override val measurementDimension: Int = measurementModel.measurementDimension

    override fun initialize(measurements: List<TimedMeasurement>): FilterState {
        require(measurements.isNotEmpty()) { "Filter $name initialization requires at least one measurement." }
        require(measurements.zipWithNext().all { (left, right) -> right.time > left.time }) {
            "Filter $name initialization measurements must be strictly increasing in time."
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

    override fun predict(state: FilterState, toTime: Double): FilterPrediction {
        require(toTime.isFinite()) { "Prediction time must be finite." }
        require(toTime >= state.estimate.time) {
            "Filter $name cannot predict backward from ${state.estimate.time} to $toTime."
        }
        requireCanonicalEstimate(state.estimate)
        val deltaTime = toTime - state.estimate.time
        val transition = motionModel.transitionMatrix(deltaTime)
        val predictedMean = transition * state.estimate.mean.value
        val predictedCovariance = (transition * state.estimate.covariance.value * transition.transpose() + motionModel.processNoise(deltaTime)).symmetrized()
        val predictedEstimate = StateEstimate(
            time = toTime,
            mean = StateVector(predictedMean),
            covariance = CovarianceMatrix(predictedCovariance),
        )
        val predictedMeasurement = expectedMeasurement(predictedEstimate)
        val h = measurementModel.measurementMatrix()
        val measurementCovariance = (h * predictedCovariance * h.transpose() + measurementModel.measurementNoise().value).symmetrized()
        return FilterPrediction(
            previousState = state,
            predictedEstimate = predictedEstimate,
            predictedMeasurement = predictedMeasurement.copy(covariance = CovarianceMatrix(measurementCovariance)),
            predictedMeasurementCovariance = CovarianceMatrix(measurementCovariance),
        )
    }

    override fun correct(
        prediction: FilterPrediction,
        measurement: TimedMeasurement,
        gatingThreshold: Double,
    ): FilterUpdate {
        require(gatingThreshold.isFinite() && gatingThreshold > 0.0) { "Gating threshold must be finite and positive." }
        require(measurement.time == prediction.predictedEstimate.time) {
            "Correction measurement time ${measurement.time} must match prediction time ${prediction.predictedEstimate.time}."
        }
        val z = Matrix.columnVector(listOf(measurement.position.x, measurement.position.y, measurement.position.z))
        val h = measurementModel.measurementMatrix()
        val r = measurementModel.measurementNoise().value
        val xPredicted = prediction.predictedEstimate.mean.value
        val pPredicted = prediction.predictedEstimate.covariance.value
        val innovation = z - h * xPredicted
        val s = (h * pPredicted * h.transpose() + r).symmetrized()
        val sInverseInnovation = s.solve(innovation)
        val mahalanobis = (innovation.transpose() * sInverseInnovation)[0, 0]
        val logLikelihood = -0.5 * (
            measurementDimension * ln(2.0 * Math.PI) +
                s.logDeterminantPositive() +
                mahalanobis
            )
        val accepted = mahalanobis <= gatingThreshold

        val updatedEstimate = if (accepted) {
            val kalmanGain = pPredicted * h.transpose() * s.inverse()
            val updatedMean = xPredicted + kalmanGain * innovation
            val identity = Matrix.identity(stateDimension)
            val residual = identity - kalmanGain * h
            val updatedCovariance = (
                residual * pPredicted * residual.transpose() +
                    kalmanGain * r * kalmanGain.transpose()
                ).symmetrized()
            StateEstimate(
                time = measurement.time,
                mean = StateVector(updatedMean),
                covariance = CovarianceMatrix(updatedCovariance),
            )
        } else {
            prediction.predictedEstimate
        }

        return FilterUpdate(
            prediction = prediction,
            updatedEstimate = updatedEstimate,
            innovation = MeasurementVector(innovation),
            innovationCovariance = CovarianceMatrix(s),
            logLikelihood = logLikelihood,
            mahalanobisDistanceSquared = mahalanobis,
            gatingThreshold = gatingThreshold,
            accepted = accepted,
        )
    }

    override fun expectedMeasurement(estimate: StateEstimate): MeasurementEstimate {
        requireCanonicalEstimate(estimate)
        val h = measurementModel.measurementMatrix()
        val covariance = (h * estimate.covariance.value * h.transpose() + measurementModel.measurementNoise().value).symmetrized()
        return MeasurementEstimate(
            time = estimate.time,
            mean = MeasurementVector(h * estimate.mean.value),
            covariance = CovarianceMatrix(covariance),
        )
    }

    private fun requireCanonicalEstimate(estimate: StateEstimate) {
        require(estimate.mean.value.rows == CanonicalKinematicState.DIMENSION) {
            "Filter $name expected canonical ${CanonicalKinematicState.DIMENSION}D state."
        }
        require(estimate.covariance.value.rows == CanonicalKinematicState.DIMENSION) {
            "Filter $name expected canonical ${CanonicalKinematicState.DIMENSION}D covariance."
        }
    }
}
