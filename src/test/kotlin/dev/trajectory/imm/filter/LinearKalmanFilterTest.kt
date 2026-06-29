package dev.trajectory.imm.filter

import dev.trajectory.imm.domain.CartesianTimedMeasurement
import dev.trajectory.imm.domain.Vector3
import dev.trajectory.imm.measurement.CartesianPositionMeasurementModel
import dev.trajectory.imm.motion.ConstantAccelerationModel9D
import dev.trajectory.imm.state.CanonicalKinematicState
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class LinearKalmanFilterTest {
    @Test
    fun `straight trajectory prediction converges to low position error`() {
        val filter = LinearKalmanFilter(
            name = "CA",
            motionModel = ConstantAccelerationModel9D(jerkSpectralDensity = 0.05),
            measurementModel = CartesianPositionMeasurementModel.isotropic(0.25),
            initializationConfig = LinearKalmanInitializationConfig(
                initialPositionVariance = 1.0,
                initialVelocityVariance = 10.0,
                initialAccelerationVariance = 10.0,
            ),
        )
        var state = filter.initialize(
            listOf(
                CartesianTimedMeasurement(0.0, Vector3(0.0, 0.0, 100.0)),
                CartesianTimedMeasurement(1.0, Vector3(10.0, 2.0, 99.0)),
                CartesianTimedMeasurement(2.0, Vector3(20.0, 4.0, 98.0)),
            ),
        )

        for (time in 3..12) {
            val t = time.toDouble()
            val prediction = filter.predict(state, t)
            val update = filter.correct(
                prediction = prediction,
                measurement = CartesianTimedMeasurement(t, Vector3(10.0 * t, 2.0 * t, 100.0 - t)),
                gatingThreshold = 100.0,
            )
            state = FilterState(update.updatedEstimate)
        }

        val finalPrediction = filter.predict(state, 13.0)
        val mean = finalPrediction.predictedEstimate.mean.value
        assertEquals(130.0, mean[CanonicalKinematicState.PX, 0], 0.5)
        assertEquals(26.0, mean[CanonicalKinematicState.PY, 0], 0.5)
        assertEquals(87.0, mean[CanonicalKinematicState.PZ, 0], 0.5)
    }

    @Test
    fun `Joseph update keeps covariance symmetric`() {
        val filter = LinearKalmanFilter(
            name = "CA",
            motionModel = ConstantAccelerationModel9D(jerkSpectralDensity = 0.5),
            measurementModel = CartesianPositionMeasurementModel.isotropic(1.0),
        )
        val state = filter.initialize(
            listOf(
                CartesianTimedMeasurement(0.0, Vector3(0.0, 0.0, 0.0)),
                CartesianTimedMeasurement(1.0, Vector3(1.0, 1.0, 1.0)),
                CartesianTimedMeasurement(2.0, Vector3(2.0, 2.0, 2.0)),
            ),
        )
        val prediction = filter.predict(state, 3.0)
        val update = filter.correct(prediction, CartesianTimedMeasurement(3.0, Vector3(3.1, 2.9, 3.0)), gatingThreshold = 100.0)

        assertTrue(update.updatedEstimate.covariance.value.isSymmetric(1.0e-10))
        assertTrue(update.updatedEstimate.covariance.value.isPositiveSemiDefinite(1.0e-8))
    }

    @Test
    fun `outlier measurement is rejected by Mahalanobis gating`() {
        val filter = LinearKalmanFilter(
            name = "CA",
            motionModel = ConstantAccelerationModel9D(jerkSpectralDensity = 0.01),
            measurementModel = CartesianPositionMeasurementModel.isotropic(0.1),
            initializationConfig = LinearKalmanInitializationConfig(
                initialPositionVariance = 0.1,
                initialVelocityVariance = 0.1,
                initialAccelerationVariance = 0.1,
            ),
        )
        val state = filter.initialize(
            listOf(
                CartesianTimedMeasurement(0.0, Vector3(0.0, 0.0, 0.0)),
                CartesianTimedMeasurement(1.0, Vector3(1.0, 0.0, 0.0)),
                CartesianTimedMeasurement(2.0, Vector3(2.0, 0.0, 0.0)),
            ),
        )
        val prediction = filter.predict(state, 3.0)
        val update = filter.correct(prediction, CartesianTimedMeasurement(3.0, Vector3(1000.0, 1000.0, 1000.0)), gatingThreshold = 7.814727903251179)

        assertFalse(update.accepted)
        assertTrue(update.mahalanobisDistanceSquared > update.gatingThreshold)
    }

    private fun assertEquals(expected: Double, actual: Double, tolerance: Double) {
        assertTrue(abs(expected - actual) <= tolerance, "Expected $expected, got $actual.")
    }
}
