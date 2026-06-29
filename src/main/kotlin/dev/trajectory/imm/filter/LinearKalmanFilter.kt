package dev.trajectory.imm.filter

import dev.trajectory.imm.domain.CovarianceMatrix
import dev.trajectory.imm.domain.CartesianTimedMeasurement
import dev.trajectory.imm.domain.MeasurementEstimate
import dev.trajectory.imm.domain.MeasurementVector
import dev.trajectory.imm.domain.StateEstimate
import dev.trajectory.imm.domain.StateVector
import dev.trajectory.imm.measurement.LinearMeasurementModel
import dev.trajectory.imm.motion.MotionModel
import dev.trajectory.imm.state.CanonicalKinematicState

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

    override fun initialize(measurements: List<CartesianTimedMeasurement>): FilterState {
        return initializeCanonicalKinematicState(name, measurements, initializationConfig)
    }

    override fun predict(state: FilterState, toTime: Double): FilterPrediction {
        require(toTime.isFinite()) { "Prediction time must be finite." }
        require(toTime >= state.estimate.time) {
            "Filter $name cannot predict backward from ${state.estimate.time} to $toTime."
        }
        requireCanonicalEstimate(name, state.estimate)
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
        measurement: CartesianTimedMeasurement,
        gatingThreshold: Double,
    ): FilterUpdate {
        require(gatingThreshold.isFinite() && gatingThreshold > 0.0) { "Gating threshold must be finite and positive." }
        require(measurement.time == prediction.predictedEstimate.time) {
            "Correction measurement time ${measurement.time} must match prediction time ${prediction.predictedEstimate.time}."
        }
        val z = measurementVector(measurement)
        val h = measurementModel.measurementMatrix()
        val r = measurementModel.measurementNoise(measurement).value
        val xPredicted = prediction.predictedEstimate.mean.value
        val pPredicted = prediction.predictedEstimate.covariance.value
        val innovation = z - h * xPredicted
        val s = (h * pPredicted * h.transpose() + r).symmetrized()
        val stats = gaussianInnovationStats(measurementDimension, innovation, s)
        val accepted = stats.mahalanobisDistanceSquared <= gatingThreshold

        val updatedEstimate = if (accepted) {
            josephUpdatedEstimate(
                time = measurement.time,
                predictedMean = xPredicted,
                predictedCovariance = pPredicted,
                measurementMatrix = h,
                measurementNoise = r,
                innovation = innovation,
                stateDimension = stateDimension,
            )
        } else {
            prediction.predictedEstimate
        }

        return FilterUpdate(
            prediction = prediction,
            updatedEstimate = updatedEstimate,
            innovation = MeasurementVector(innovation),
            innovationCovariance = CovarianceMatrix(s),
            logLikelihood = stats.logLikelihood,
            mahalanobisDistanceSquared = stats.mahalanobisDistanceSquared,
            gatingThreshold = gatingThreshold,
            accepted = accepted,
        )
    }

    override fun expectedMeasurement(estimate: StateEstimate): MeasurementEstimate {
        requireCanonicalEstimate(name, estimate)
        return expectedLinearMeasurement(measurementModel, estimate)
    }
}
