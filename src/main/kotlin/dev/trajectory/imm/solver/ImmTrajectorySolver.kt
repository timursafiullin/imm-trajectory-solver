package dev.trajectory.imm.solver

import dev.trajectory.imm.coordinate.CartesianCoordinateFrameAdapter
import dev.trajectory.imm.coordinate.CartesianSolverFrame
import dev.trajectory.imm.coordinate.SolverFrame
import dev.trajectory.imm.coordinate.Wgs84CoordinateFrameAdapter
import dev.trajectory.imm.coordinate.Wgs84FrameConfig
import dev.trajectory.imm.coordinate.Wgs84SolverFrame
import dev.trajectory.imm.domain.CartesianPositionCovariance
import dev.trajectory.imm.domain.CartesianTimedMeasurement
import dev.trajectory.imm.domain.CovarianceMatrix
import dev.trajectory.imm.domain.EnuPositionNoise
import dev.trajectory.imm.domain.MeasurementEstimate
import dev.trajectory.imm.domain.MeasurementVector
import dev.trajectory.imm.domain.PositionMeasurementNoise
import dev.trajectory.imm.domain.StateEstimate
import dev.trajectory.imm.domain.StateVector
import dev.trajectory.imm.domain.TrajectoryPrediction
import dev.trajectory.imm.domain.TrajectoryUpdate
import dev.trajectory.imm.domain.Vector3
import dev.trajectory.imm.domain.Wgs84Position
import dev.trajectory.imm.domain.Wgs84TimedMeasurement
import dev.trajectory.imm.filter.FilterState
import dev.trajectory.imm.filter.FilterUpdate
import dev.trajectory.imm.filter.LinearKalmanFilter
import dev.trajectory.imm.filter.LinearKalmanInitializationConfig
import dev.trajectory.imm.filter.TrackingFilter
import dev.trajectory.imm.math.Matrix
import dev.trajectory.imm.measurement.CartesianPositionMeasurementModel
import dev.trajectory.imm.motion.ConstantAccelerationModel9D
import dev.trajectory.imm.motion.ConstantVelocityModel9D
import dev.trajectory.imm.motion.SingerAccelerationModel9D
import dev.trajectory.imm.state.CanonicalKinematicState
import dev.trajectory.imm.validation.Validators
import dev.trajectory.imm.validation.logSumExp
import dev.trajectory.imm.validation.normalizeProbabilities
import kotlin.math.exp
import kotlin.math.ln

data class ImmSolverConfig(
    val measurementNoise: PositionMeasurementNoise = EnuPositionNoise.isotropic(25.0),
    val cvAccelerationSpectralDensity: Double = 0.5,
    val caJerkSpectralDensity: Double = 1.0,
    val singerManeuverTime: Double = 8.0,
    val singerAccelerationNoiseIntensity: Double = 2.0,
    val gatingThreshold: Double = MetricImmTrajectoryCore.DEFAULT_GATING_THRESHOLD,
) {
    init {
        require(cvAccelerationSpectralDensity.isFinite() && cvAccelerationSpectralDensity >= 0.0) {
            "CV acceleration spectral density must be finite and non-negative."
        }
        require(caJerkSpectralDensity.isFinite() && caJerkSpectralDensity >= 0.0) {
            "CA jerk spectral density must be finite and non-negative."
        }
        require(singerManeuverTime.isFinite() && singerManeuverTime > 0.0) {
            "Singer maneuver time must be finite and positive."
        }
        require(singerAccelerationNoiseIntensity.isFinite() && singerAccelerationNoiseIntensity >= 0.0) {
            "Singer acceleration noise intensity must be finite and non-negative."
        }
        require(gatingThreshold.isFinite() && gatingThreshold > 0.0) {
            "Gating threshold must be finite and positive."
        }
    }

    companion object {
        fun withIsotropicEnuMeasurementVariance(
            variance: Double,
            cvAccelerationSpectralDensity: Double = 0.5,
            caJerkSpectralDensity: Double = 1.0,
            singerManeuverTime: Double = 8.0,
            singerAccelerationNoiseIntensity: Double = 2.0,
            gatingThreshold: Double = MetricImmTrajectoryCore.DEFAULT_GATING_THRESHOLD,
        ): ImmSolverConfig {
            return ImmSolverConfig(
                measurementNoise = EnuPositionNoise.isotropic(variance),
                cvAccelerationSpectralDensity = cvAccelerationSpectralDensity,
                caJerkSpectralDensity = caJerkSpectralDensity,
                singerManeuverTime = singerManeuverTime,
                singerAccelerationNoiseIntensity = singerAccelerationNoiseIntensity,
                gatingThreshold = gatingThreshold,
            )
        }
    }
}

class CartesianImmTrajectorySolver(
    private val core: MetricImmTrajectoryCore,
    private val adapter: CartesianCoordinateFrameAdapter,
) : TrajectoryPredictionBackend<CartesianTimedMeasurement, Vector3> {
    override fun initialize(measurements: List<CartesianTimedMeasurement>): SolverState {
        return core.initialize(measurements.map(adapter::toInternalMeasurement))
    }

    override fun predictNext(
        state: SolverState,
        toTime: Double,
    ): TrajectoryPrediction<Vector3> {
        return adapter.fromInternalPrediction(core.predictNext(state, toTime))
    }

    override fun update(
        state: SolverState,
        measurement: CartesianTimedMeasurement,
    ): SolverStep<Vector3> {
        val step = core.update(state, adapter.toInternalMeasurement(measurement))
        return SolverStep(
            previousState = step.previousState,
            nextState = step.nextState,
            update = adapter.fromInternalUpdate(step.update),
        )
    }

    override fun predictHorizon(
        state: SolverState,
        futureTimes: List<Double>,
    ): List<TrajectoryPrediction<Vector3>> {
        return core.predictHorizon(state, futureTimes).map(adapter::fromInternalPrediction)
    }

    override fun reset(): SolverState = core.reset()
}

class Wgs84ImmTrajectorySolver(
    private val core: MetricImmTrajectoryCore,
    private val adapter: Wgs84CoordinateFrameAdapter,
) : TrajectoryPredictionBackend<Wgs84TimedMeasurement, Wgs84Position> {
    override fun initialize(measurements: List<Wgs84TimedMeasurement>): SolverState {
        return core.initialize(measurements.map(adapter::toInternalMeasurement))
    }

    override fun predictNext(
        state: SolverState,
        toTime: Double,
    ): TrajectoryPrediction<Wgs84Position> {
        return adapter.fromInternalPrediction(core.predictNext(state, toTime))
    }

    override fun update(
        state: SolverState,
        measurement: Wgs84TimedMeasurement,
    ): SolverStep<Wgs84Position> {
        val step = core.update(state, adapter.toInternalMeasurement(measurement))
        return SolverStep(
            previousState = step.previousState,
            nextState = step.nextState,
            update = adapter.fromInternalUpdate(step.update),
        )
    }

    override fun predictHorizon(
        state: SolverState,
        futureTimes: List<Double>,
    ): List<TrajectoryPrediction<Wgs84Position>> {
        return core.predictHorizon(state, futureTimes).map(adapter::fromInternalPrediction)
    }

    override fun reset(): SolverState = core.reset()
}

class MetricImmTrajectoryCore(
    private val filters: List<TrackingFilter>,
    private val transitionMatrix: Matrix,
    initialProbabilities: Map<String, Double>,
    private val gatingThreshold: Double = DEFAULT_GATING_THRESHOLD,
    private val frame: SolverFrame = CartesianSolverFrame,
) : TrajectoryPredictionBackend<CartesianTimedMeasurement, Vector3> {
    private val names: List<String> = filters.map { it.name }
    private val initialProbabilities: Map<String, Double> = normalizeProbabilities(initialProbabilities)

    init {
        require(filters.isNotEmpty()) { "IMM core requires at least one filter." }
        require(names.toSet().size == names.size) { "IMM filter names must be unique, got $names." }
        require(gatingThreshold.isFinite() && gatingThreshold > 0.0) { "Gating threshold must be finite and positive." }
        filters.forEach { filter ->
            Validators.requireCanonicalDimension("Filter ${filter.name}", filter.stateDimension, CanonicalKinematicState.DIMENSION)
        }
        Validators.requireTransitionMatrix(transitionMatrix, filters.size)
        Validators.requireProbabilities(this.initialProbabilities, names)
    }

    override fun initialize(measurements: List<CartesianTimedMeasurement>): SolverState {
        require(measurements.isNotEmpty()) { "IMM initialization requires at least one measurement." }
        require(measurements.zipWithNext().all { (left, right) -> right.time > left.time }) {
            "IMM initialization measurements must be strictly increasing in time."
        }
        val filterStates = filters.associate { filter ->
            val state = filter.initialize(measurements)
            Validators.requireCovariance(state.estimate.covariance.value, "Initial ${filter.name}")
            filter.name to state
        }
        return SolverState(
            time = measurements.last().time,
            filterStates = filterStates,
            modelProbabilities = initialProbabilities,
            frame = frame,
        )
    }

    override fun predictNext(state: SolverState, toTime: Double): TrajectoryPrediction<Vector3> {
        requireValidState(state)
        require(toTime.isFinite()) { "Prediction time must be finite." }
        require(toTime >= state.time) { "IMM cannot predict backward from ${state.time} to $toTime." }
        val predictions = filters.associate { filter ->
            filter.name to filter.predict(state.filterStates.getValue(filter.name), toTime)
        }
        return combinePrediction(
            time = toTime,
            weights = state.modelProbabilities,
            estimates = predictions.mapValues { (_, prediction) -> prediction.predictedEstimate },
            measurementEstimates = predictions.mapValues { (_, prediction) -> prediction.predictedMeasurement },
            metadataPrefix = "predict",
        )
    }

    override fun update(state: SolverState, measurement: CartesianTimedMeasurement): SolverStep<Vector3> {
        requireValidState(state)
        require(measurement.time > state.time) {
            "IMM update measurement time ${measurement.time} must be strictly greater than state time ${state.time}."
        }
        val mixed = mixStates(state, measurement.time)
        val priorPrediction = combinePrediction(
            time = measurement.time,
            weights = mixed.priorProbabilities,
            estimates = mixed.predictions.mapValues { (_, prediction) -> prediction.predictedEstimate },
            measurementEstimates = mixed.predictions.mapValues { (_, prediction) -> prediction.predictedMeasurement },
            metadataPrefix = "prior",
        )
        val updates = filters.associate { filter ->
            val prediction = mixed.predictions.getValue(filter.name)
            filter.name to filter.correct(prediction, measurement, gatingThreshold)
        }
        val posteriorProbabilities = updateModelProbabilities(mixed.priorProbabilities, updates)
        val nextFilterStates = updates.mapValues { (name, update) ->
            Validators.requireCovariance(update.updatedEstimate.covariance.value, "Updated $name")
            FilterState(update.updatedEstimate)
        }
        val combinedEstimate = combineEstimates(
            weights = posteriorProbabilities,
            estimates = updates.mapValues { (_, update) -> update.updatedEstimate },
            time = measurement.time,
        )
        val accepted = updates.values.any { it.accepted }
        val logEvidence = logSumExp(names.map { name -> updates.getValue(name).logLikelihood + ln(mixed.priorProbabilities.getValue(name)) })
        val bestMahalanobis = updates.values.minOf { it.mahalanobisDistanceSquared }
        val nextState = SolverState(
            time = measurement.time,
            filterStates = nextFilterStates,
            modelProbabilities = posteriorProbabilities,
            frame = state.frame,
        )
        val metadata = buildMap {
            names.forEach { name ->
                val update = updates.getValue(name)
                put("prior.$name", mixed.priorProbabilities.getValue(name))
                put("posterior.$name", posteriorProbabilities.getValue(name))
                put("logLikelihood.$name", update.logLikelihood)
                put("mahalanobis.$name", update.mahalanobisDistanceSquared)
                put("accepted.$name", if (update.accepted) 1.0 else 0.0)
            }
        }
        val trajectoryUpdate = TrajectoryUpdate(
            time = measurement.time,
            prediction = priorPrediction,
            correctedPosition = vector3FromState(combinedEstimate),
            internalUpdatedEstimate = combinedEstimate,
            accepted = accepted,
            logLikelihood = logEvidence,
            mahalanobisDistanceSquared = bestMahalanobis,
            modelProbabilities = posteriorProbabilities,
            metadata = metadata,
        )
        requireValidState(nextState)
        return SolverStep(
            previousState = state,
            nextState = nextState,
            update = trajectoryUpdate,
        )
    }

    override fun predictHorizon(
        state: SolverState,
        futureTimes: List<Double>,
    ): List<TrajectoryPrediction<Vector3>> {
        requireValidState(state)
        require(futureTimes.zipWithNext().all { (left, right) -> right > left }) {
            "Future prediction times must be strictly increasing."
        }
        var temporaryState = state
        return futureTimes.map { time ->
            val prediction = predictNext(temporaryState, time)
            val nextFilterStates = filters.associate { filter ->
                val predicted = filter.predict(temporaryState.filterStates.getValue(filter.name), time)
                filter.name to FilterState(predicted.predictedEstimate)
            }
            temporaryState = SolverState(
                time = time,
                filterStates = nextFilterStates,
                modelProbabilities = temporaryState.modelProbabilities,
                frame = temporaryState.frame,
            )
            prediction
        }
    }

    override fun reset(): SolverState {
        return SolverState(
            time = Double.NaN,
            filterStates = emptyMap(),
            modelProbabilities = emptyMap(),
            frame = frame,
        )
    }

    private fun mixStates(state: SolverState, toTime: Double): MixedPrediction {
        val priorProbabilities = names.associateWith { destination ->
            val destinationIndex = names.indexOf(destination)
            names.indices.sumOf { sourceIndex ->
                transitionMatrix[sourceIndex, destinationIndex] * state.modelProbabilities.getValue(names[sourceIndex])
            }
        }
        Validators.requireProbabilities(priorProbabilities, names)

        val predictions = filters.associate { destinationFilter ->
            val destinationIndex = names.indexOf(destinationFilter.name)
            val weights = names.associateWith { source ->
                val sourceIndex = names.indexOf(source)
                transitionMatrix[sourceIndex, destinationIndex] *
                    state.modelProbabilities.getValue(source) /
                    priorProbabilities.getValue(destinationFilter.name)
            }
            val mixedEstimate = combineEstimates(
                weights = weights,
                estimates = state.filterStates.mapValues { (_, filterState) -> filterState.estimate },
                time = state.time,
            )
            destinationFilter.name to destinationFilter.predict(FilterState(mixedEstimate), toTime)
        }
        return MixedPrediction(
            priorProbabilities = priorProbabilities,
            predictions = predictions,
        )
    }

    private fun combinePrediction(
        time: Double,
        weights: Map<String, Double>,
        estimates: Map<String, StateEstimate>,
        measurementEstimates: Map<String, MeasurementEstimate>,
        metadataPrefix: String,
    ): TrajectoryPrediction<Vector3> {
        val combinedState = combineEstimates(weights, estimates, time)
        val combinedMeasurement = combineMeasurements(weights, measurementEstimates, time)
        val metadata = buildMap {
            names.forEach { name -> put("$metadataPrefix.probability.$name", weights.getValue(name)) }
        }
        return TrajectoryPrediction(
            time = time,
            expectedPosition = vector3FromMeasurement(combinedMeasurement),
            covariance = CartesianPositionCovariance(combinedMeasurement.covariance),
            internalStateEstimate = combinedState,
            modelProbabilities = weights,
            metadata = metadata,
        )
    }

    private fun combineEstimates(
        weights: Map<String, Double>,
        estimates: Map<String, StateEstimate>,
        time: Double,
    ): StateEstimate {
        Validators.requireProbabilities(weights, names, tolerance = 1.0e-7)
        require(estimates.keys == names.toSet()) { "Estimate names ${estimates.keys} must match IMM names ${names.toSet()}." }
        var mean = Matrix.zeros(CanonicalKinematicState.DIMENSION, 1)
        names.forEach { name ->
            val estimate = estimates.getValue(name)
            require(estimate.mean.value.rows == CanonicalKinematicState.DIMENSION) {
                "Estimate $name must be canonical ${CanonicalKinematicState.DIMENSION}D."
            }
            mean += estimate.mean.value * weights.getValue(name)
        }
        var covariance = Matrix.zeros(CanonicalKinematicState.DIMENSION, CanonicalKinematicState.DIMENSION)
        names.forEach { name ->
            val estimate = estimates.getValue(name)
            val delta = estimate.mean.value - mean
            covariance += (estimate.covariance.value + Matrix.outerProduct(delta, delta)) * weights.getValue(name)
        }
        val symmetrized = covariance.symmetrized()
        Validators.requireCovariance(symmetrized, "Combined IMM")
        return StateEstimate(
            time = time,
            mean = StateVector(mean),
            covariance = CovarianceMatrix(symmetrized),
        )
    }

    private fun combineMeasurements(
        weights: Map<String, Double>,
        measurementEstimates: Map<String, MeasurementEstimate>,
        time: Double,
    ): MeasurementEstimate {
        require(measurementEstimates.keys == names.toSet()) {
            "Measurement estimate names ${measurementEstimates.keys} must match IMM names ${names.toSet()}."
        }
        val dimension = measurementEstimates.values.first().mean.value.rows
        var mean = Matrix.zeros(dimension, 1)
        names.forEach { name ->
            mean += measurementEstimates.getValue(name).mean.value * weights.getValue(name)
        }
        var covariance = Matrix.zeros(dimension, dimension)
        names.forEach { name ->
            val estimate = measurementEstimates.getValue(name)
            val delta = estimate.mean.value - mean
            covariance += (estimate.covariance.value + Matrix.outerProduct(delta, delta)) * weights.getValue(name)
        }
        val symmetrized = covariance.symmetrized()
        Validators.requireCovariance(symmetrized, "Combined measurement")
        return MeasurementEstimate(
            time = time,
            mean = MeasurementVector(mean),
            covariance = CovarianceMatrix(symmetrized),
        )
    }

    private fun updateModelProbabilities(
        priorProbabilities: Map<String, Double>,
        updates: Map<String, FilterUpdate>,
    ): Map<String, Double> {
        val logWeights = names.associateWith { name ->
            updates.getValue(name).logLikelihood + ln(priorProbabilities.getValue(name))
        }
        val normalizer = logSumExp(names.map { logWeights.getValue(it) })
        val probabilities = names.associateWith { name -> exp(logWeights.getValue(name) - normalizer) }
        Validators.requireProbabilities(probabilities, names, tolerance = 1.0e-7)
        return probabilities
    }

    private fun requireValidState(state: SolverState) {
        require(state.time.isFinite()) { "Solver state is not initialized." }
        require(state.frame.id == frame.id) {
            "Solver state frame ${state.frame.id} does not match solver frame ${frame.id}."
        }
        require(state.filterStates.keys == names.toSet()) {
            "Solver state filter names ${state.filterStates.keys} must match IMM names ${names.toSet()}."
        }
        Validators.requireProbabilities(state.modelProbabilities, names, tolerance = 1.0e-7)
        state.filterStates.forEach { (name, filterState) ->
            require(filterState.estimate.time == state.time) {
                "Filter state $name time ${filterState.estimate.time} must match solver time ${state.time}."
            }
            require(filterState.estimate.mean.value.rows == CanonicalKinematicState.DIMENSION) {
                "Filter state $name must be canonical ${CanonicalKinematicState.DIMENSION}D."
            }
            Validators.requireCovariance(filterState.estimate.covariance.value, "Filter state $name")
        }
    }

    private fun vector3FromMeasurement(measurement: MeasurementEstimate): Vector3 {
        val mean = measurement.mean.value
        require(mean.rows == 3 && mean.columns == 1) {
            "Expected 3x1 measurement mean, got ${mean.rows}x${mean.columns}."
        }
        return Vector3(mean[0, 0], mean[1, 0], mean[2, 0])
    }

    private fun vector3FromState(estimate: StateEstimate): Vector3 {
        val mean = estimate.mean.value
        require(mean.rows >= 3 && mean.columns == 1) {
            "Expected at least 3x1 state mean, got ${mean.rows}x${mean.columns}."
        }
        return Vector3(mean[0, 0], mean[1, 0], mean[2, 0])
    }

    private data class MixedPrediction(
        val priorProbabilities: Map<String, Double>,
        val predictions: Map<String, dev.trajectory.imm.filter.FilterPrediction>,
    )

    companion object {
        const val DEFAULT_GATING_THRESHOLD: Double = 7.814727903251179
    }
}

object ImmTrajectorySolver {
    fun cartesian(config: ImmSolverConfig = ImmSolverConfig()): CartesianImmTrajectorySolver {
        val measurementNoise = config.measurementNoise
        require(measurementNoise is EnuPositionNoise) {
            "Cartesian solver requires ENU/Cartesian measurement noise."
        }
        val adapter = CartesianCoordinateFrameAdapter(measurementNoise)
        return CartesianImmTrajectorySolver(
            core = createCore(
                config = config,
                frame = CartesianSolverFrame,
                baseMeasurementNoise = measurementNoise.covariance,
            ),
            adapter = adapter,
        )
    }

    fun wgs84(
        origin: Wgs84Position,
        config: ImmSolverConfig = ImmSolverConfig(),
        frameConfig: Wgs84FrameConfig = Wgs84FrameConfig(origin = origin),
    ): Wgs84ImmTrajectorySolver {
        require(frameConfig.origin == origin) {
            "WGS84 frame config origin must match factory origin."
        }
        val frame = Wgs84SolverFrame.fromConfig(frameConfig)
        val adapter = Wgs84CoordinateFrameAdapter(frame, config.measurementNoise)
        return Wgs84ImmTrajectorySolver(
            core = createCore(
                config = config,
                frame = frame,
                baseMeasurementNoise = adapter.measurementNoiseAt(origin),
            ),
            adapter = adapter,
        )
    }

    private fun createCore(
        config: ImmSolverConfig,
        frame: SolverFrame,
        baseMeasurementNoise: CovarianceMatrix,
    ): MetricImmTrajectoryCore {
        val measurementModel = CartesianPositionMeasurementModel(baseMeasurementNoise)
        val init = LinearKalmanInitializationConfig()
        val filters = listOf(
            LinearKalmanFilter(
                name = "CV",
                motionModel = ConstantVelocityModel9D(accelerationSpectralDensity = config.cvAccelerationSpectralDensity),
                measurementModel = measurementModel,
                initializationConfig = init,
            ),
            LinearKalmanFilter(
                name = "CA",
                motionModel = ConstantAccelerationModel9D(config.caJerkSpectralDensity),
                measurementModel = measurementModel,
                initializationConfig = init,
            ),
            LinearKalmanFilter(
                name = "Singer",
                motionModel = SingerAccelerationModel9D(config.singerManeuverTime, config.singerAccelerationNoiseIntensity),
                measurementModel = measurementModel,
                initializationConfig = init,
            ),
        )
        val transition = Matrix.ofRows(
            listOf(
                listOf(0.95, 0.04, 0.01),
                listOf(0.03, 0.94, 0.03),
                listOf(0.02, 0.08, 0.90),
            ),
        )
        val initialProbabilities = mapOf("CV" to 1.0 / 3.0, "CA" to 1.0 / 3.0, "Singer" to 1.0 / 3.0)
        return MetricImmTrajectoryCore(
            filters = filters,
            transitionMatrix = transition,
            initialProbabilities = initialProbabilities,
            gatingThreshold = config.gatingThreshold,
            frame = frame,
        )
    }
}
