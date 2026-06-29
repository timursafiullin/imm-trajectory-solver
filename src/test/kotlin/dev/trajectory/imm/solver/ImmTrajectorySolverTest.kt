package dev.trajectory.imm.solver

import dev.trajectory.imm.coordinate.Wgs84FrameConfig
import dev.trajectory.imm.coordinate.Wgs84Geodesy
import dev.trajectory.imm.coordinate.Wgs84SolverFrame
import dev.trajectory.imm.domain.CartesianTimedMeasurement
import dev.trajectory.imm.domain.CovarianceMatrix
import dev.trajectory.imm.domain.EnuPositionNoise
import dev.trajectory.imm.domain.Vector3
import dev.trajectory.imm.domain.Wgs84Position
import dev.trajectory.imm.domain.Wgs84PositionCovariance
import dev.trajectory.imm.domain.Wgs84PositionNoise
import dev.trajectory.imm.domain.Wgs84TimedMeasurement
import dev.trajectory.imm.math.Matrix
import dev.trajectory.imm.state.CanonicalKinematicState
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue

class ImmTrajectorySolverTest {
    @Test
    fun `cartesian solver initializes canonical 9D filters and normalized probabilities`() {
        val solver = ImmTrajectorySolver.cartesian(ImmSolverConfig.withIsotropicEnuMeasurementVariance(1.0))
        val state = solver.initialize(cartesianHistory())

        assertEquals(3, state.filterStates.size)
        state.filterStates.values.forEach { filterState ->
            assertEquals(CanonicalKinematicState.DIMENSION, filterState.estimate.mean.value.rows)
            assertEquals(CanonicalKinematicState.DIMENSION, filterState.estimate.covariance.value.rows)
        }
        assertEquals(1.0, state.modelProbabilities.values.sum(), 1.0e-12)
    }

    @Test
    fun `cartesian predict next and horizon do not mutate online state`() {
        val solver = ImmTrajectorySolver.cartesian(ImmSolverConfig.withIsotropicEnuMeasurementVariance(1.0))
        val state = solver.initialize(cartesianHistory())
        val before = state.copy()

        solver.predictNext(state, 3.0)
        solver.predictHorizon(state, listOf(3.0, 4.0, 5.0))

        assertEquals(before, state)
    }

    @Test
    fun `cartesian updates keep probabilities normalized and return diagnostics`() {
        val solver = ImmTrajectorySolver.cartesian(ImmSolverConfig.withIsotropicEnuMeasurementVariance(1.0))
        var state = solver.initialize(cartesianHistory())

        val step = solver.update(state, CartesianTimedMeasurement(3.0, Vector3(3.0, 0.0, 0.0)))
        state = step.nextState

        assertTrue(step.update.accepted)
        assertTrue(step.update.mahalanobisDistanceSquared >= 0.0)
        assertTrue(step.update.logLikelihood.isFinite())
        assertEquals(1.0, state.modelProbabilities.values.sum(), 1.0e-10)
    }

    @Test
    fun `accelerated segment shifts probability away from pure CV`() {
        val solver = ImmTrajectorySolver.cartesian(
            ImmSolverConfig.withIsotropicEnuMeasurementVariance(
                variance = 0.25,
                cvAccelerationSpectralDensity = 0.05,
                caJerkSpectralDensity = 0.5,
                singerAccelerationNoiseIntensity = 1.0,
                gatingThreshold = 1.0e6,
            ),
        )
        var state = solver.initialize(
            listOf(
                CartesianTimedMeasurement(0.0, acceleratedPoint(0.0)),
                CartesianTimedMeasurement(1.0, acceleratedPoint(1.0)),
                CartesianTimedMeasurement(2.0, acceleratedPoint(2.0)),
            ),
        )

        for (time in 3..12) {
            val step = solver.update(state, CartesianTimedMeasurement(time.toDouble(), acceleratedPoint(time.toDouble())))
            state = step.nextState
        }

        val cv = state.modelProbabilities.getValue("CV")
        val accelerationAware = state.modelProbabilities.getValue("CA") + state.modelProbabilities.getValue("Singer")
        assertTrue(accelerationAware > cv, "Expected CA+Singer probability $accelerationAware to exceed CV probability $cv.")
    }

    @Test
    fun `WGS84 solver initializes with explicit frame and returns WGS84 predictions`() {
        val history = wgs84History()
        val origin = history.first().position
        val solver = ImmTrajectorySolver.wgs84(
            origin = origin,
            config = ImmSolverConfig.withIsotropicEnuMeasurementVariance(1.0),
        )

        val state = solver.initialize(history)
        val prediction = solver.predictNext(state, 3.0)

        val frame = assertIs<Wgs84SolverFrame>(state.frame)
        assertEquals(origin, frame.origin)
        assertEquals(CanonicalKinematicState.DIMENSION, prediction.internalStateEstimate.mean.value.rows)
        assertTrue(prediction.expectedPosition.latitudeDegrees.isFinite())
        assertTrue(prediction.expectedPosition.longitudeDegrees.isFinite())
        assertTrue(prediction.expectedPosition.heightMeters.isFinite())
    }

    @Test
    fun `WGS84 solver matches equivalent local ENU run internally`() {
        val wgs84History = wgs84History()
        val wgs84Next = Wgs84TimedMeasurement(
            time = 3.0,
            position = Wgs84Position(
                latitudeDegrees = 56.8389270,
                longitudeDegrees = 60.6057650,
                heightMeters = 263.0,
            ),
        )
        val origin = wgs84History.first().position
        val cartesianHistory = wgs84History.map { measurement -> measurement.toLocalCartesian(origin) }
        val cartesianNext = wgs84Next.toLocalCartesian(origin)
        val config = ImmSolverConfig.withIsotropicEnuMeasurementVariance(
            variance = 1.0,
            gatingThreshold = 1.0e9,
        )
        val wgs84Solver = ImmTrajectorySolver.wgs84(origin = origin, config = config)
        val cartesianSolver = ImmTrajectorySolver.cartesian(config)

        val wgs84State = wgs84Solver.initialize(wgs84History)
        val cartesianState = cartesianSolver.initialize(cartesianHistory)
        val wgs84Prediction = wgs84Solver.predictNext(wgs84State, 3.0)
        val cartesianPrediction = cartesianSolver.predictNext(cartesianState, 3.0)
        val wgs84Step = wgs84Solver.update(wgs84State, wgs84Next)
        val cartesianStep = cartesianSolver.update(cartesianState, cartesianNext)

        assertMatrixClose(
            expected = cartesianPrediction.internalStateEstimate.mean.value,
            actual = wgs84Prediction.internalStateEstimate.mean.value,
            tolerance = 1.0e-9,
        )
        assertMatrixClose(
            expected = cartesianStep.update.internalUpdatedEstimate.mean.value,
            actual = wgs84Step.update.internalUpdatedEstimate.mean.value,
            tolerance = 1.0e-9,
        )
        cartesianStep.update.modelProbabilities.forEach { (name, probability) ->
            assertEquals(probability, wgs84Step.update.modelProbabilities.getValue(name), 1.0e-12)
        }
    }

    @Test
    fun `WGS84 predict next and horizon do not mutate online state`() {
        val history = wgs84History()
        val solver = ImmTrajectorySolver.wgs84(
            origin = history.first().position,
            config = ImmSolverConfig.withIsotropicEnuMeasurementVariance(1.0),
        )
        val state = solver.initialize(history)
        val before = state.copy()

        solver.predictNext(state, 3.0)
        solver.predictHorizon(state, listOf(3.0, 4.0, 5.0))

        assertEquals(before, state)
    }

    @Test
    fun `solvers reject states from incompatible frames`() {
        val history = wgs84History()
        val wgs84Solver = ImmTrajectorySolver.wgs84(
            origin = history.first().position,
            config = ImmSolverConfig.withIsotropicEnuMeasurementVariance(1.0),
        )
        val otherWgs84Solver = ImmTrajectorySolver.wgs84(
            origin = history.first().position.copy(latitudeDegrees = history.first().position.latitudeDegrees + 0.01),
            config = ImmSolverConfig.withIsotropicEnuMeasurementVariance(1.0),
        )
        val cartesianSolver = ImmTrajectorySolver.cartesian(ImmSolverConfig.withIsotropicEnuMeasurementVariance(1.0))
        val wgs84State = wgs84Solver.initialize(history)
        val cartesianState = cartesianSolver.initialize(cartesianHistory())

        assertFailsWith<IllegalArgumentException> {
            otherWgs84Solver.predictNext(wgs84State, 3.0)
        }
        assertFailsWith<IllegalArgumentException> {
            wgs84Solver.predictNext(cartesianState, 3.0)
        }
        assertFailsWith<IllegalArgumentException> {
            cartesianSolver.predictNext(wgs84State, 3.0)
        }
    }

    @Test
    fun `WGS84 solver rejects measurements outside valid local radius`() {
        val history = wgs84History()
        val solver = ImmTrajectorySolver.wgs84(
            origin = history.first().position,
            config = ImmSolverConfig.withIsotropicEnuMeasurementVariance(1.0),
            frameConfig = Wgs84FrameConfig(
                origin = history.first().position,
                validRadiusMeters = 1.0,
            ),
        )

        assertFailsWith<IllegalArgumentException> {
            solver.initialize(history)
        }
    }

    @Test
    fun `WGS84 prediction exposes ENU and geodetic covariance`() {
        val history = wgs84History()
        val origin = history.first().position
        val geodeticNoise = Wgs84PositionNoise(
            covariance = CovarianceMatrix(Matrix.diagonal(listOf(1.0e-10, 1.0e-10, 1.0))),
        )
        val solver = ImmTrajectorySolver.wgs84(
            origin = origin,
            config = ImmSolverConfig(measurementNoise = geodeticNoise),
        )

        val state = solver.initialize(history)
        val covariance = assertIs<Wgs84PositionCovariance>(solver.predictNext(state, 3.0).covariance)

        assertEquals(3, covariance.enuCovariance.value.rows)
        assertEquals(3, covariance.geodeticCovariance.value.rows)
        assertTrue(covariance.enuCovariance.value.isPositiveSemiDefinite(1.0e-7))
        assertTrue(covariance.geodeticCovariance.value.isPositiveSemiDefinite(1.0e-7))
    }

    @Test
    fun `cartesian solver supports mixed Kalman filter set`() {
        val solver = ImmTrajectorySolver.cartesian(
            config = ImmSolverConfig.withIsotropicEnuMeasurementVariance(
                variance = 1.0,
                gatingThreshold = 1.0e6,
            ),
            filterSet = mixedCvCaSingerFilterSet(),
        )
        val state = solver.initialize(cartesianHistory())
        val prediction = solver.predictNext(state, 3.0)
        val step = solver.update(state, CartesianTimedMeasurement(3.0, Vector3(3.0, 0.0, 0.0)))

        assertEquals(setOf("CV", "CA", "Singer"), state.modelProbabilities.keys)
        assertEquals(setOf("CV", "CA", "Singer"), prediction.modelProbabilities.keys)
        assertEquals(setOf("CV", "CA", "Singer"), step.update.modelProbabilities.keys)
        assertTrue(step.update.accepted)
        assertEquals(CanonicalKinematicState.DIMENSION, prediction.internalStateEstimate.mean.value.rows)
    }

    @Test
    fun `WGS84 solver supports mixed Kalman filter set`() {
        val history = wgs84History()
        val origin = history.first().position
        val solver = ImmTrajectorySolver.wgs84(
            origin = origin,
            config = ImmSolverConfig.withIsotropicEnuMeasurementVariance(
                variance = 1.0,
                gatingThreshold = 1.0e6,
            ),
            filterSet = mixedCvCaSingerFilterSet(),
        )
        val state = solver.initialize(history)
        val prediction = solver.predictNext(state, 3.0)

        assertEquals(setOf("CV", "CA", "Singer"), state.modelProbabilities.keys)
        assertEquals(setOf("CV", "CA", "Singer"), prediction.modelProbabilities.keys)
        assertTrue(prediction.expectedPosition.latitudeDegrees.isFinite())
        assertTrue(prediction.expectedPosition.longitudeDegrees.isFinite())
        assertTrue(prediction.expectedPosition.heightMeters.isFinite())
    }

    @Test
    fun `non-default filter set requires explicit transition matrix`() {
        val config = ImmSolverConfig.withIsotropicEnuMeasurementVariance(1.0)
        val filterSet = ImmFilterSetSpec.of(
            ImmModelSpec("OnlyCV", MotionModelType.CV, KalmanFilterType.LINEAR),
        )

        assertFailsWith<IllegalArgumentException> {
            ImmTrajectorySolver.cartesian(
                config = config,
                filterSet = filterSet,
            ).initialize(cartesianHistory())
        }
    }

    @Test
    fun `custom single-model filter set works with explicit transition matrix`() {
        val solver = ImmTrajectorySolver.cartesian(
            config = ImmSolverConfig.withIsotropicEnuMeasurementVariance(1.0),
            filterSet = ImmFilterSetSpec(
                models = listOf(
                    ImmModelSpec("OnlyCV", MotionModelType.CV, KalmanFilterType.LINEAR),
                ),
                transitionMatrix = Matrix.ofRows(listOf(listOf(1.0))),
                initialProbabilities = mapOf("OnlyCV" to 1.0),
            ),
        )
        val state = solver.initialize(cartesianHistory())

        assertEquals(setOf("OnlyCV"), state.modelProbabilities.keys)
        assertEquals(1.0, state.modelProbabilities.getValue("OnlyCV"), 1.0e-12)
    }

    @Test
    fun `invalid explicit transition matrix is rejected`() {
        val filterSet = ImmFilterSetSpec(
            models = listOf(
                ImmModelSpec("OnlyCV", MotionModelType.CV, KalmanFilterType.LINEAR),
            ),
            transitionMatrix = Matrix.ofRows(listOf(listOf(0.5))),
        )

        assertFailsWith<IllegalArgumentException> {
            ImmTrajectorySolver.cartesian(
                config = ImmSolverConfig.withIsotropicEnuMeasurementVariance(1.0),
                filterSet = filterSet,
            ).initialize(cartesianHistory())
        }
    }

    @Test
    fun `explicit initial probabilities are normalized by core`() {
        val solver = ImmTrajectorySolver.cartesian(
            config = ImmSolverConfig.withIsotropicEnuMeasurementVariance(1.0),
            filterSet = ImmFilterSetSpec(
                models = listOf(
                    ImmModelSpec("CV", MotionModelType.CV, KalmanFilterType.LINEAR),
                    ImmModelSpec("CA", MotionModelType.CA, KalmanFilterType.LINEAR),
                    ImmModelSpec("Singer", MotionModelType.SINGER, KalmanFilterType.LINEAR),
                ),
                initialProbabilities = mapOf(
                    "CV" to 2.0,
                    "CA" to 1.0,
                    "Singer" to 1.0,
                ),
            ),
        )
        val state = solver.initialize(cartesianHistory())

        assertEquals(0.5, state.modelProbabilities.getValue("CV"), 1.0e-12)
        assertEquals(0.25, state.modelProbabilities.getValue("CA"), 1.0e-12)
        assertEquals(0.25, state.modelProbabilities.getValue("Singer"), 1.0e-12)
    }

    private fun cartesianHistory(): List<CartesianTimedMeasurement> {
        return listOf(
            CartesianTimedMeasurement(0.0, Vector3(0.0, 0.0, 0.0)),
            CartesianTimedMeasurement(1.0, Vector3(1.0, 0.0, 0.0)),
            CartesianTimedMeasurement(2.0, Vector3(2.0, 0.0, 0.0)),
        )
    }

    private fun mixedCvCaSingerFilterSet(): ImmFilterSetSpec {
        return ImmFilterSetSpec(
            models = listOf(
                ImmModelSpec("CV", MotionModelType.CV, KalmanFilterType.LINEAR),
                ImmModelSpec("CA", MotionModelType.CA, KalmanFilterType.INNOVATION_ADAPTIVE),
                ImmModelSpec("Singer", MotionModelType.SINGER, KalmanFilterType.EXTENDED),
            ),
        )
    }

    private fun acceleratedPoint(time: Double): Vector3 {
        return Vector3(
            x = 0.5 * 3.0 * time * time,
            y = 2.0 * time,
            z = 100.0 + 0.25 * time * time,
        )
    }

    private fun wgs84History(): List<Wgs84TimedMeasurement> {
        return listOf(
            Wgs84TimedMeasurement(
                time = 0.0,
                position = Wgs84Position(
                    latitudeDegrees = 56.8389000,
                    longitudeDegrees = 60.6057000,
                    heightMeters = 250.0,
                ),
            ),
            Wgs84TimedMeasurement(
                time = 1.0,
                position = Wgs84Position(
                    latitudeDegrees = 56.8389090,
                    longitudeDegrees = 60.6057210,
                    heightMeters = 254.0,
                ),
            ),
            Wgs84TimedMeasurement(
                time = 2.0,
                position = Wgs84Position(
                    latitudeDegrees = 56.8389180,
                    longitudeDegrees = 60.6057430,
                    heightMeters = 258.5,
                ),
            ),
        )
    }

    private fun Wgs84TimedMeasurement.toLocalCartesian(origin: Wgs84Position): CartesianTimedMeasurement {
        return CartesianTimedMeasurement(
            time = time,
            position = Wgs84Geodesy.geodeticToLocalEnu(
                position = position,
                origin = origin,
            ),
        )
    }

    private fun assertMatrixClose(
        expected: Matrix,
        actual: Matrix,
        tolerance: Double,
    ) {
        assertEquals(expected.rows, actual.rows)
        assertEquals(expected.columns, actual.columns)
        for (row in 0 until expected.rows) {
            for (column in 0 until expected.columns) {
                assertEquals(expected[row, column], actual[row, column], tolerance)
            }
        }
    }

    private fun assertEquals(expected: Double, actual: Double, tolerance: Double) {
        assertTrue(abs(expected - actual) <= tolerance, "Expected $expected, got $actual.")
    }
}
