package dev.trajectory.imm.filter

import dev.trajectory.imm.domain.TimedMeasurement
import dev.trajectory.imm.domain.Vector3
import dev.trajectory.imm.measurement.CartesianPositionMeasurementModel
import dev.trajectory.imm.motion.ConstantAccelerationModel9D
import dev.trajectory.imm.motion.LinearizedMotionModel
import dev.trajectory.imm.state.CanonicalKinematicState
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

class AdditionalKalmanFiltersTest {
    @Test
    fun `extended Kalman filter matches linear Kalman filter for linearized CA model`() {
        val motionModel = ConstantAccelerationModel9D(jerkSpectralDensity = 0.2)
        val measurementModel = CartesianPositionMeasurementModel.isotropic(0.5)
        val initializationConfig = LinearKalmanInitializationConfig(
            initialPositionVariance = 2.0,
            initialVelocityVariance = 5.0,
            initialAccelerationVariance = 10.0,
        )
        val linear = LinearKalmanFilter(
            name = "linear-ca",
            motionModel = motionModel,
            measurementModel = measurementModel,
            initializationConfig = initializationConfig,
        )
        val ekf = ExtendedKalmanFilter(
            name = "ekf-ca",
            motionModel = LinearizedMotionModel(motionModel),
            measurementModel = measurementModel,
            initializationConfig = initializationConfig,
        )
        val history = listOf(
            TimedMeasurement(0.0, Vector3(0.0, 0.0, 100.0)),
            TimedMeasurement(1.0, Vector3(6.0, -1.0, 101.0)),
            TimedMeasurement(2.0, Vector3(14.0, -2.0, 104.0)),
        )

        val linearPrediction = linear.predict(linear.initialize(history), 3.0)
        val ekfPrediction = ekf.predict(ekf.initialize(history), 3.0)
        val measurement = TimedMeasurement(3.0, Vector3(24.0, -3.0, 109.0))
        val linearUpdate = linear.correct(linearPrediction, measurement, gatingThreshold = 1000.0)
        val ekfUpdate = ekf.correct(ekfPrediction, measurement, gatingThreshold = 1000.0)

        for (index in 0 until CanonicalKinematicState.DIMENSION) {
            assertClose(
                linearUpdate.updatedEstimate.mean.value[index, 0],
                ekfUpdate.updatedEstimate.mean.value[index, 0],
                1.0e-9,
            )
        }
        assertClose(linearUpdate.logLikelihood, ekfUpdate.logLikelihood, 1.0e-9)
        assertClose(
            linearUpdate.updatedEstimate.covariance.value.trace(),
            ekfUpdate.updatedEstimate.covariance.value.trace(),
            1.0e-9,
        )
    }

    @Test
    fun `fading memory filter inflates prediction covariance before correction`() {
        val motionModel = ConstantAccelerationModel9D(jerkSpectralDensity = 0.1)
        val measurementModel = CartesianPositionMeasurementModel.isotropic(1.0)
        val initializationConfig = LinearKalmanInitializationConfig(
            initialPositionVariance = 1.0,
            initialVelocityVariance = 2.0,
            initialAccelerationVariance = 3.0,
        )
        val linear = LinearKalmanFilter(
            name = "linear-ca",
            motionModel = motionModel,
            measurementModel = measurementModel,
            initializationConfig = initializationConfig,
        )
        val fading = FadingMemoryKalmanFilter(
            name = "fading-ca",
            motionModel = motionModel,
            measurementModel = measurementModel,
            fadingFactor = 1.20,
            initializationConfig = initializationConfig,
        )
        val history = straightHistory()

        val linearPrediction = linear.predict(linear.initialize(history), 3.0)
        val fadingPrediction = fading.predict(fading.initialize(history), 3.0)

        assertTrue(
            fadingPrediction.predictedEstimate.covariance.value.trace() >
                linearPrediction.predictedEstimate.covariance.value.trace(),
            "Fading-memory covariance must be larger than ordinary linear prediction covariance.",
        )
        assertTrue(fadingPrediction.predictedEstimate.covariance.value.isPositiveSemiDefinite(1.0e-8))
    }

    @Test
    fun `innovation adaptive filter inflates measurement noise for unexpectedly large residuals`() {
        val motionModel = ConstantAccelerationModel9D(jerkSpectralDensity = 0.01)
        val measurementModel = CartesianPositionMeasurementModel.isotropic(0.1)
        val initializationConfig = LinearKalmanInitializationConfig(
            initialPositionVariance = 0.1,
            initialVelocityVariance = 0.1,
            initialAccelerationVariance = 0.1,
        )
        val linear = LinearKalmanFilter(
            name = "linear-ca",
            motionModel = motionModel,
            measurementModel = measurementModel,
            initializationConfig = initializationConfig,
        )
        val adaptive = InnovationAdaptiveKalmanFilter(
            name = "adaptive-ca",
            motionModel = motionModel,
            measurementModel = measurementModel,
            adaptiveConfig = InnovationAdaptiveKalmanConfig(
                targetMahalanobisDistanceSquared = 3.0,
                adaptationRate = 1.0,
                maxMeasurementNoiseScale = 100.0,
            ),
            initializationConfig = initializationConfig,
        )
        val history = straightHistory()
        val measurement = TimedMeasurement(3.0, Vector3(8.0, 0.0, 0.0))

        val linearUpdate = linear.correct(linear.predict(linear.initialize(history), 3.0), measurement, gatingThreshold = 1000.0)
        val adaptiveUpdate = adaptive.correct(adaptive.predict(adaptive.initialize(history), 3.0), measurement, gatingThreshold = 1000.0)

        assertTrue(
            adaptiveUpdate.innovationCovariance.value.trace() > linearUpdate.innovationCovariance.value.trace(),
            "Adaptive filter must enlarge innovation covariance when residual is unexpectedly large.",
        )
        assertTrue(
            adaptiveUpdate.mahalanobisDistanceSquared < linearUpdate.mahalanobisDistanceSquared,
            "Adaptive Mahalanobis distance must drop after measurement-noise inflation.",
        )
        assertTrue(adaptiveUpdate.updatedEstimate.covariance.value.isPositiveSemiDefinite(1.0e-8))
    }

    private fun straightHistory(): List<TimedMeasurement> {
        return listOf(
            TimedMeasurement(0.0, Vector3(0.0, 0.0, 0.0)),
            TimedMeasurement(1.0, Vector3(1.0, 0.0, 0.0)),
            TimedMeasurement(2.0, Vector3(2.0, 0.0, 0.0)),
        )
    }

    private fun assertClose(
        expected: Double,
        actual: Double,
        tolerance: Double,
    ) {
        assertTrue(abs(expected - actual) <= tolerance, "Expected $expected, got $actual.")
    }
}
