package dev.trajectory.imm.filter

import dev.trajectory.imm.domain.CovarianceMatrix
import dev.trajectory.imm.domain.CartesianTimedMeasurement
import dev.trajectory.imm.domain.MeasurementEstimate
import dev.trajectory.imm.domain.MeasurementVector
import dev.trajectory.imm.domain.StateEstimate
import dev.trajectory.imm.domain.StateVector
import dev.trajectory.imm.measurement.EkfMeasurementModel
import dev.trajectory.imm.motion.NonlinearMotionModel
import dev.trajectory.imm.state.CanonicalKinematicState

class ExtendedKalmanFilter(
    override val name: String,
    private val motionModel: NonlinearMotionModel,
    private val measurementModel: EkfMeasurementModel,
    private val initializationConfig: LinearKalmanInitializationConfig = LinearKalmanInitializationConfig(),
) : TrackingFilter {
    init {
        require(name.isNotBlank()) { "Filter name must not be blank." }
        require(motionModel.stateDimension == CanonicalKinematicState.DIMENSION) {
            "Extended Kalman filter must expose canonical ${CanonicalKinematicState.DIMENSION}D state."
        }
        require(measurementModel.measurementDimension == 3) {
            "Extended Kalman filter expects 3D Cartesian measurements in the current backend version."
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
        val transitionJacobian = motionModel.jacobian(state.estimate.mean.value, deltaTime)
        val predictedMean = motionModel.propagate(state.estimate.mean.value, deltaTime)
        val predictedCovariance = (
            transitionJacobian * state.estimate.covariance.value * transitionJacobian.transpose() +
                motionModel.processNoise(deltaTime)
            ).symmetrized()
        val predictedEstimate = StateEstimate(
            time = toTime,
            mean = StateVector(predictedMean),
            covariance = CovarianceMatrix(predictedCovariance),
        )
        val predictedMeasurement = expectedMeasurement(predictedEstimate)
        val h = measurementModel.jacobian(predictedEstimate.mean)
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
        val predictedState = prediction.predictedEstimate.mean
        val predictedMeasurement = measurementModel.predictMeasurement(predictedState, measurement.time)
        val h = measurementModel.jacobian(predictedState)
        val r = measurementModel.measurementNoise(measurement).value
        val innovation = measurementVector(measurement) - predictedMeasurement.mean.value
        val s = (h * prediction.predictedEstimate.covariance.value * h.transpose() + r).symmetrized()
        val stats = gaussianInnovationStats(measurementDimension, innovation, s)
        val accepted = stats.mahalanobisDistanceSquared <= gatingThreshold
        val updatedEstimate = if (accepted) {
            josephUpdatedEstimate(
                time = measurement.time,
                predictedMean = prediction.predictedEstimate.mean.value,
                predictedCovariance = prediction.predictedEstimate.covariance.value,
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
        val predicted = measurementModel.predictMeasurement(estimate.mean, estimate.time)
        val h = measurementModel.jacobian(estimate.mean)
        val covariance = (h * estimate.covariance.value * h.transpose() + measurementModel.measurementNoise().value).symmetrized()
        return predicted.copy(covariance = CovarianceMatrix(covariance))
    }
}
