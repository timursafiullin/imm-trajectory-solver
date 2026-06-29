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

class FadingMemoryKalmanFilter(
    override val name: String,
    private val motionModel: MotionModel,
    private val measurementModel: LinearMeasurementModel,
    private val fadingFactor: Double = 1.05,
    private val initializationConfig: LinearKalmanInitializationConfig = LinearKalmanInitializationConfig(),
) : TrackingFilter {
    init {
        require(name.isNotBlank()) { "Filter name must not be blank." }
        require(fadingFactor.isFinite() && fadingFactor >= 1.0) { "Fading factor must be finite and at least 1.0." }
        require(motionModel.stateDimension == CanonicalKinematicState.DIMENSION) {
            "Fading-memory Kalman filter must expose canonical ${CanonicalKinematicState.DIMENSION}D state."
        }
        require(measurementModel.measurementDimension == 3) {
            "Fading-memory Kalman filter expects 3D Cartesian measurements."
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
        val propagatedCovariance = transition * state.estimate.covariance.value * transition.transpose()
        val predictedCovariance = (propagatedCovariance * fadingFactor + motionModel.processNoise(deltaTime)).symmetrized()
        val predictedEstimate = StateEstimate(
            time = toTime,
            mean = StateVector(predictedMean),
            covariance = CovarianceMatrix(predictedCovariance),
        )
        val h = measurementModel.measurementMatrix()
        val measurementCovariance = (h * predictedCovariance * h.transpose() + measurementModel.measurementNoise().value).symmetrized()
        val predictedMeasurement = expectedMeasurement(predictedEstimate)
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
        val h = measurementModel.measurementMatrix()
        val r = measurementModel.measurementNoise(measurement).value
        val innovation = measurementVector(measurement) - h * prediction.predictedEstimate.mean.value
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
        return expectedLinearMeasurement(measurementModel, estimate)
    }
}
