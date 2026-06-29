package dev.trajectory.imm.solver

import dev.trajectory.imm.filter.InnovationAdaptiveKalmanConfig
import dev.trajectory.imm.math.Matrix

enum class MotionModelType {
    CV,
    CA,
    SINGER,
}

enum class KalmanFilterType {
    LINEAR,
    EXTENDED,
    INNOVATION_ADAPTIVE,
    FADING_MEMORY,
}

data class ImmModelSpec(
    val name: String,
    val motionModelType: MotionModelType,
    val filterType: KalmanFilterType,
    val cvAccelerationSpectralDensity: Double? = null,
    val caJerkSpectralDensity: Double? = null,
    val singerManeuverTime: Double? = null,
    val singerAccelerationNoiseIntensity: Double? = null,
    val adaptiveConfig: InnovationAdaptiveKalmanConfig = InnovationAdaptiveKalmanConfig(),
    val fadingFactor: Double = 1.05,
) {
    init {
        require(name.isNotBlank()) { "IMM model name must not be blank." }
        require(cvAccelerationSpectralDensity == null || cvAccelerationSpectralDensity.isFinite() && cvAccelerationSpectralDensity >= 0.0) {
            "CV acceleration spectral density override must be finite and non-negative."
        }
        require(caJerkSpectralDensity == null || caJerkSpectralDensity.isFinite() && caJerkSpectralDensity >= 0.0) {
            "CA jerk spectral density override must be finite and non-negative."
        }
        require(singerManeuverTime == null || singerManeuverTime.isFinite() && singerManeuverTime > 0.0) {
            "Singer maneuver time override must be finite and positive."
        }
        require(singerAccelerationNoiseIntensity == null || singerAccelerationNoiseIntensity.isFinite() && singerAccelerationNoiseIntensity >= 0.0) {
            "Singer acceleration noise intensity override must be finite and non-negative."
        }
        require(fadingFactor.isFinite() && fadingFactor >= 1.0) {
            "Fading factor must be finite and at least 1.0."
        }
    }
}

data class ImmFilterSetSpec(
    val models: List<ImmModelSpec>,
    val transitionMatrix: Matrix? = null,
    val initialProbabilities: Map<String, Double>? = null,
) {
    init {
        require(models.isNotEmpty()) { "IMM filter set must contain at least one model." }
        val names = models.map { it.name }
        require(names.toSet().size == names.size) { "IMM model names must be unique, got $names." }
        initialProbabilities?.let { probabilities ->
            require(probabilities.keys == names.toSet()) {
                "Initial probability names ${probabilities.keys} must match IMM model names ${names.toSet()}."
            }
            require(probabilities.values.all { it.isFinite() && it >= 0.0 }) {
                "Initial probabilities must be finite and non-negative."
            }
            require(probabilities.values.sum() > 0.0) {
                "At least one initial probability must be positive."
            }
        }
    }

    companion object {
        fun defaultCvCaSinger(): ImmFilterSetSpec {
            return ImmFilterSetSpec(
                models = listOf(
                    ImmModelSpec("CV", MotionModelType.CV, KalmanFilterType.LINEAR),
                    ImmModelSpec("CA", MotionModelType.CA, KalmanFilterType.LINEAR),
                    ImmModelSpec("Singer", MotionModelType.SINGER, KalmanFilterType.LINEAR),
                ),
            )
        }

        fun of(vararg models: ImmModelSpec): ImmFilterSetSpec {
            return ImmFilterSetSpec(models = models.toList())
        }
    }
}
