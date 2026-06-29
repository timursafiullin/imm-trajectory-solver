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

data class InnovationAdaptiveKalmanConfig(
    val targetMahalanobisDistanceSquared: Double = 3.0,
    val adaptationRate: Double = 0.65,
    val maxMeasurementNoiseScale: Double = 50.0,
) {
    init {
        require(targetMahalanobisDistanceSquared.isFinite() && targetMahalanobisDistanceSquared > 0.0) {
            "Target Mahalanobis distance must be finite and positive."
        }
        require(adaptationRate.isFinite() && adaptationRate in 0.0..1.0) {
            "Adaptation rate must be finite and in [0, 1]."
        }
        require(maxMeasurementNoiseScale.isFinite() && maxMeasurementNoiseScale >= 1.0) {
            "Maximum measurement noise scale must be finite and at least 1.0."
        }
    }
}

class InnovationAdaptiveKalmanFilter(
    override val name: String,
    private val motionModel: MotionModel,
    private val measurementModel: LinearMeasurementModel,
    private val adaptiveConfig: InnovationAdaptiveKalmanConfig = InnovationAdaptiveKalmanConfig(),
    private val initializationConfig: LinearKalmanInitializationConfig = LinearKalmanInitializationConfig(),
) : TrackingFilter {
    init {
        require(name.isNotBlank()) { "Filter name must not be blank." }
        require(motionModel.stateDimension == CanonicalKinematicState.DIMENSION) {
            "Adaptive Kalman filter must expose canonical ${CanonicalKinematicState.DIMENSION}D state."
        }
        require(measurementModel.measurementDimension == 3) {
            "Adaptive Kalman filter expects 3D Cartesian measurements."
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
        val baseMeasurementNoise = measurementModel.measurementNoise(measurement).value
        val predictedMean = prediction.predictedEstimate.mean.value
        val predictedCovariance = prediction.predictedEstimate.covariance.value
        val innovation = measurementVector(measurement) - h * predictedMean
        val baseInnovationCovariance = (h * predictedCovariance * h.transpose() + baseMeasurementNoise).symmetrized()
        val baseStats = gaussianInnovationStats(measurementDimension, innovation, baseInnovationCovariance)
        val ratio = baseStats.mahalanobisDistanceSquared / adaptiveConfig.targetMahalanobisDistanceSquared
        val measurementNoiseScale = if (ratio <= 1.0) {
            1.0
        } else {
            (1.0 + adaptiveConfig.adaptationRate * (ratio - 1.0)).coerceIn(1.0, adaptiveConfig.maxMeasurementNoiseScale)
        }
        val adaptedMeasurementNoise = baseMeasurementNoise * measurementNoiseScale
        val adaptedInnovationCovariance = (h * predictedCovariance * h.transpose() + adaptedMeasurementNoise).symmetrized()
        val adaptedStats = gaussianInnovationStats(measurementDimension, innovation, adaptedInnovationCovariance)
        val accepted = adaptedStats.mahalanobisDistanceSquared <= gatingThreshold
        val updatedEstimate = if (accepted) {
            josephUpdatedEstimate(
                time = measurement.time,
                predictedMean = predictedMean,
                predictedCovariance = predictedCovariance,
                measurementMatrix = h,
                measurementNoise = adaptedMeasurementNoise,
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
            innovationCovariance = CovarianceMatrix(adaptedInnovationCovariance),
            logLikelihood = adaptedStats.logLikelihood,
            mahalanobisDistanceSquared = adaptedStats.mahalanobisDistanceSquared,
            gatingThreshold = gatingThreshold,
            accepted = accepted,
        )
    }

    override fun expectedMeasurement(estimate: StateEstimate): MeasurementEstimate {
        requireCanonicalEstimate(name, estimate)
        return expectedLinearMeasurement(measurementModel, estimate)
    }
}
