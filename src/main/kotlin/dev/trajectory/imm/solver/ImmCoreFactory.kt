package dev.trajectory.imm.solver

import dev.trajectory.imm.coordinate.SolverFrame
import dev.trajectory.imm.domain.CovarianceMatrix
import dev.trajectory.imm.filter.ExtendedKalmanFilter
import dev.trajectory.imm.filter.FadingMemoryKalmanFilter
import dev.trajectory.imm.filter.InnovationAdaptiveKalmanFilter
import dev.trajectory.imm.filter.LinearKalmanFilter
import dev.trajectory.imm.filter.LinearKalmanInitializationConfig
import dev.trajectory.imm.filter.TrackingFilter
import dev.trajectory.imm.math.Matrix
import dev.trajectory.imm.measurement.CartesianPositionMeasurementModel
import dev.trajectory.imm.motion.ConstantAccelerationModel9D
import dev.trajectory.imm.motion.ConstantVelocityModel9D
import dev.trajectory.imm.motion.LinearizedMotionModel
import dev.trajectory.imm.motion.MotionModel
import dev.trajectory.imm.motion.SingerAccelerationModel9D

object ImmCoreFactory {
    private val defaultModelNames: List<String> = listOf("CV", "CA", "Singer")

    fun createCore(
        config: ImmSolverConfig,
        frame: SolverFrame,
        baseMeasurementNoise: CovarianceMatrix,
        filterSet: ImmFilterSetSpec,
    ): MetricImmTrajectoryCore {
        val measurementModel = CartesianPositionMeasurementModel(baseMeasurementNoise)
        val initializationConfig = LinearKalmanInitializationConfig()
        val filters = filterSet.models.map { model ->
            createFilter(
                model = model,
                config = config,
                measurementModel = measurementModel,
                initializationConfig = initializationConfig,
            )
        }
        return MetricImmTrajectoryCore(
            filters = filters,
            transitionMatrix = transitionMatrix(filterSet),
            initialProbabilities = initialProbabilities(filterSet),
            gatingThreshold = config.gatingThreshold,
            frame = frame,
        )
    }

    private fun createFilter(
        model: ImmModelSpec,
        config: ImmSolverConfig,
        measurementModel: CartesianPositionMeasurementModel,
        initializationConfig: LinearKalmanInitializationConfig,
    ): TrackingFilter {
        val motionModel = createMotionModel(model, config)
        return when (model.filterType) {
            KalmanFilterType.LINEAR -> LinearKalmanFilter(
                name = model.name,
                motionModel = motionModel,
                measurementModel = measurementModel,
                initializationConfig = initializationConfig,
            )

            KalmanFilterType.EXTENDED -> ExtendedKalmanFilter(
                name = model.name,
                motionModel = LinearizedMotionModel(motionModel),
                measurementModel = measurementModel,
                initializationConfig = initializationConfig,
            )

            KalmanFilterType.INNOVATION_ADAPTIVE -> InnovationAdaptiveKalmanFilter(
                name = model.name,
                motionModel = motionModel,
                measurementModel = measurementModel,
                adaptiveConfig = model.adaptiveConfig,
                initializationConfig = initializationConfig,
            )

            KalmanFilterType.FADING_MEMORY -> FadingMemoryKalmanFilter(
                name = model.name,
                motionModel = motionModel,
                measurementModel = measurementModel,
                fadingFactor = model.fadingFactor,
                initializationConfig = initializationConfig,
            )
        }
    }

    private fun createMotionModel(
        model: ImmModelSpec,
        config: ImmSolverConfig,
    ): MotionModel {
        return when (model.motionModelType) {
            MotionModelType.CV -> ConstantVelocityModel9D(
                accelerationSpectralDensity = model.cvAccelerationSpectralDensity
                    ?: config.cvAccelerationSpectralDensity,
            )

            MotionModelType.CA -> ConstantAccelerationModel9D(
                jerkSpectralDensity = model.caJerkSpectralDensity
                    ?: config.caJerkSpectralDensity,
            )

            MotionModelType.SINGER -> SingerAccelerationModel9D(
                maneuverTime = model.singerManeuverTime
                    ?: config.singerManeuverTime,
                accelerationNoiseIntensity = model.singerAccelerationNoiseIntensity
                    ?: config.singerAccelerationNoiseIntensity,
            )
        }
    }

    private fun transitionMatrix(filterSet: ImmFilterSetSpec): Matrix {
        filterSet.transitionMatrix?.let { return it }
        require(isDefaultCvCaSinger(filterSet)) {
            "IMM transition matrix is required for non-default model sets. " +
                "Only CV, CA, Singer in this exact order can use the default transition matrix."
        }
        return defaultTransitionMatrix()
    }

    private fun initialProbabilities(filterSet: ImmFilterSetSpec): Map<String, Double> {
        filterSet.initialProbabilities?.let { return it }
        val probability = 1.0 / filterSet.models.size
        return filterSet.models.associate { it.name to probability }
    }

    private fun isDefaultCvCaSinger(filterSet: ImmFilterSetSpec): Boolean {
        if (filterSet.models.map { it.name } != defaultModelNames) {
            return false
        }
        return filterSet.models.map { it.motionModelType } == listOf(
            MotionModelType.CV,
            MotionModelType.CA,
            MotionModelType.SINGER,
        )
    }

    fun defaultTransitionMatrix(): Matrix {
        return Matrix.ofRows(
            listOf(
                listOf(0.95, 0.04, 0.01),
                listOf(0.03, 0.94, 0.03),
                listOf(0.02, 0.08, 0.90),
            ),
        )
    }
}
