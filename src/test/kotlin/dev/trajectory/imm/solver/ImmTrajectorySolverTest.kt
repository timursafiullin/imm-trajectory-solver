package dev.trajectory.imm.solver

import dev.trajectory.imm.domain.TimedMeasurement
import dev.trajectory.imm.domain.Vector3
import dev.trajectory.imm.state.CanonicalKinematicState
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ImmTrajectorySolverTest {
    @Test
    fun `default solver initializes canonical 9D filters and normalized probabilities`() {
        val solver = ImmTrajectorySolver.defaultCartesian(measurementVariance = 1.0)
        val state = solver.initialize(
            listOf(
                TimedMeasurement(0.0, Vector3(0.0, 0.0, 0.0)),
                TimedMeasurement(1.0, Vector3(1.0, 0.0, 0.0)),
                TimedMeasurement(2.0, Vector3(2.0, 0.0, 0.0)),
            ),
        )

        assertEquals(3, state.filterStates.size)
        state.filterStates.values.forEach { filterState ->
            assertEquals(CanonicalKinematicState.DIMENSION, filterState.estimate.mean.value.rows)
            assertEquals(CanonicalKinematicState.DIMENSION, filterState.estimate.covariance.value.rows)
        }
        assertEquals(1.0, state.modelProbabilities.values.sum(), 1.0e-12)
    }

    @Test
    fun `predict next and horizon do not mutate online state`() {
        val solver = ImmTrajectorySolver.defaultCartesian(measurementVariance = 1.0)
        val state = solver.initialize(
            listOf(
                TimedMeasurement(0.0, Vector3(0.0, 0.0, 0.0)),
                TimedMeasurement(1.0, Vector3(1.0, 1.0, 0.0)),
                TimedMeasurement(2.0, Vector3(2.0, 2.0, 0.0)),
            ),
        )
        val before = state.copy()

        solver.predictNext(state, 3.0)
        solver.predictHorizon(state, listOf(3.0, 4.0, 5.0))

        assertEquals(before, state)
    }

    @Test
    fun `updates keep probabilities normalized and return observable gating diagnostics`() {
        val solver = ImmTrajectorySolver.defaultCartesian(measurementVariance = 1.0)
        var state = solver.initialize(
            listOf(
                TimedMeasurement(0.0, Vector3(0.0, 0.0, 0.0)),
                TimedMeasurement(1.0, Vector3(1.0, 0.0, 0.0)),
                TimedMeasurement(2.0, Vector3(2.0, 0.0, 0.0)),
            ),
        )

        val step = solver.update(state, TimedMeasurement(3.0, Vector3(3.0, 0.0, 0.0)))
        state = step.nextState

        assertTrue(step.update.accepted)
        assertTrue(step.update.mahalanobisDistanceSquared >= 0.0)
        assertTrue(step.update.logLikelihood.isFinite())
        assertEquals(1.0, state.modelProbabilities.values.sum(), 1.0e-10)
    }

    @Test
    fun `accelerated segment shifts probability away from pure CV`() {
        val solver = ImmTrajectorySolver.defaultCartesian(
            measurementVariance = 0.25,
            cvAccelerationSpectralDensity = 0.05,
            caJerkSpectralDensity = 0.5,
            singerAccelerationNoiseIntensity = 1.0,
            gatingThreshold = 1.0e6,
        )
        var state = solver.initialize(
            listOf(
                TimedMeasurement(0.0, acceleratedPoint(0.0)),
                TimedMeasurement(1.0, acceleratedPoint(1.0)),
                TimedMeasurement(2.0, acceleratedPoint(2.0)),
            ),
        )

        for (time in 3..12) {
            val step = solver.update(state, TimedMeasurement(time.toDouble(), acceleratedPoint(time.toDouble())))
            state = step.nextState
        }

        val cv = state.modelProbabilities.getValue("CV")
        val accelerationAware = state.modelProbabilities.getValue("CA") + state.modelProbabilities.getValue("Singer")
        assertTrue(accelerationAware > cv, "Expected CA+Singer probability $accelerationAware to exceed CV probability $cv.")
    }

    private fun acceleratedPoint(time: Double): Vector3 {
        return Vector3(
            x = 0.5 * 3.0 * time * time,
            y = 2.0 * time,
            z = 100.0 + 0.25 * time * time,
        )
    }

    private fun assertEquals(expected: Double, actual: Double, tolerance: Double) {
        assertTrue(abs(expected - actual) <= tolerance, "Expected $expected, got $actual.")
    }
}
