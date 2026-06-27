package dev.trajectory.imm.motion

import dev.trajectory.imm.math.Matrix
import dev.trajectory.imm.state.CanonicalKinematicState
import kotlin.math.exp
import kotlin.math.max

interface MotionModel {
    val name: String
    val stateDimension: Int

    fun transitionMatrix(deltaTime: Double): Matrix

    fun processNoise(deltaTime: Double): Matrix

    fun propagate(mean: Matrix, deltaTime: Double): Matrix {
        require(mean.rows == stateDimension && mean.columns == 1) {
            "Motion model $name expected ${stateDimension}x1 state, got ${mean.rows}x${mean.columns}."
        }
        return transitionMatrix(deltaTime) * mean
    }
}

class ConstantVelocityModel9D(
    private val accelerationDamping: Double = 0.0,
    private val accelerationSpectralDensity: Double = 1.0,
    private val accelerationSlotSpectralDensity: Double = 25.0,
) : MotionModel {
    init {
        require(accelerationDamping in 0.0..1.0) { "Acceleration damping must be in [0, 1]." }
        require(accelerationSpectralDensity >= 0.0 && accelerationSpectralDensity.isFinite()) {
            "Acceleration spectral density must be finite and non-negative."
        }
        require(accelerationSlotSpectralDensity >= 0.0 && accelerationSlotSpectralDensity.isFinite()) {
            "Acceleration slot spectral density must be finite and non-negative."
        }
    }

    override val name: String = "CV"
    override val stateDimension: Int = CanonicalKinematicState.DIMENSION

    override fun transitionMatrix(deltaTime: Double): Matrix {
        requireValidDeltaTime(deltaTime)
        var matrix = Matrix.identity(stateDimension)
        matrix = matrix.with(CanonicalKinematicState.PX, CanonicalKinematicState.VX, deltaTime)
        matrix = matrix.with(CanonicalKinematicState.PY, CanonicalKinematicState.VY, deltaTime)
        matrix = matrix.with(CanonicalKinematicState.PZ, CanonicalKinematicState.VZ, deltaTime)
        matrix = matrix.with(CanonicalKinematicState.AX, CanonicalKinematicState.AX, accelerationDamping)
        matrix = matrix.with(CanonicalKinematicState.AY, CanonicalKinematicState.AY, accelerationDamping)
        matrix = matrix.with(CanonicalKinematicState.AZ, CanonicalKinematicState.AZ, accelerationDamping)
        return matrix
    }

    override fun processNoise(deltaTime: Double): Matrix {
        requireValidDeltaTime(deltaTime)
        var q = Matrix.zeros(stateDimension, stateDimension)
        val dt2 = deltaTime * deltaTime
        val dt3 = dt2 * deltaTime
        val dt4 = dt2 * dt2
        val positionVariance = accelerationSpectralDensity * dt4 / 4.0
        val positionVelocityCovariance = accelerationSpectralDensity * dt3 / 2.0
        val velocityVariance = accelerationSpectralDensity * dt2
        for ((p, v) in canonicalAxes()) {
            q = q.with(p, p, positionVariance)
            q = q.with(p, v, positionVelocityCovariance)
            q = q.with(v, p, positionVelocityCovariance)
            q = q.with(v, v, velocityVariance)
        }
        val accelerationVariance = accelerationSlotSpectralDensity * max(deltaTime, 1.0e-9)
        for (a in CanonicalKinematicState.ACCELERATION_INDICES) {
            q = q.with(a, a, accelerationVariance)
        }
        return q
    }
}

class ConstantAccelerationModel9D(
    private val jerkSpectralDensity: Double = 1.0,
) : MotionModel {
    init {
        require(jerkSpectralDensity >= 0.0 && jerkSpectralDensity.isFinite()) {
            "Jerk spectral density must be finite and non-negative."
        }
    }

    override val name: String = "CA"
    override val stateDimension: Int = CanonicalKinematicState.DIMENSION

    override fun transitionMatrix(deltaTime: Double): Matrix {
        requireValidDeltaTime(deltaTime)
        var matrix = Matrix.identity(stateDimension)
        val halfDt2 = 0.5 * deltaTime * deltaTime
        for ((p, v, a) in canonicalTriples()) {
            matrix = matrix.with(p, v, deltaTime)
            matrix = matrix.with(p, a, halfDt2)
            matrix = matrix.with(v, a, deltaTime)
        }
        return matrix
    }

    override fun processNoise(deltaTime: Double): Matrix {
        requireValidDeltaTime(deltaTime)
        return whiteJerkProcessNoise(deltaTime, jerkSpectralDensity)
    }
}

class SingerAccelerationModel9D(
    maneuverTime: Double = 8.0,
    private val accelerationNoiseIntensity: Double = 2.0,
) : MotionModel {
    private val alpha: Double = 1.0 / maneuverTime

    init {
        require(maneuverTime.isFinite() && maneuverTime > 0.0) { "Singer maneuver time must be finite and positive." }
        require(accelerationNoiseIntensity.isFinite() && accelerationNoiseIntensity >= 0.0) {
            "Singer acceleration noise intensity must be finite and non-negative."
        }
    }

    override val name: String = "Singer"
    override val stateDimension: Int = CanonicalKinematicState.DIMENSION

    override fun transitionMatrix(deltaTime: Double): Matrix {
        requireValidDeltaTime(deltaTime)
        if (alpha * deltaTime < 1.0e-6) {
            return ConstantAccelerationModel9D(accelerationNoiseIntensity).transitionMatrix(deltaTime)
        }
        val phi = exp(-alpha * deltaTime)
        val positionAcceleration = (alpha * deltaTime - 1.0 + phi) / (alpha * alpha)
        val velocityAcceleration = (1.0 - phi) / alpha
        var matrix = Matrix.identity(stateDimension)
        for ((p, v, a) in canonicalTriples()) {
            matrix = matrix.with(p, v, deltaTime)
            matrix = matrix.with(p, a, positionAcceleration)
            matrix = matrix.with(v, a, velocityAcceleration)
            matrix = matrix.with(a, a, phi)
        }
        return matrix
    }

    override fun processNoise(deltaTime: Double): Matrix {
        requireValidDeltaTime(deltaTime)
        val base = whiteJerkProcessNoise(deltaTime, accelerationNoiseIntensity)
        val accelerationMemoryVariance = accelerationNoiseIntensity * (1.0 - exp(-2.0 * alpha * deltaTime)) / (2.0 * alpha)
        var q = base
        for (a in CanonicalKinematicState.ACCELERATION_INDICES) {
            q = q.with(a, a, q[a, a] + accelerationMemoryVariance)
        }
        return q.symmetrized()
    }
}

private fun requireValidDeltaTime(deltaTime: Double) {
    require(deltaTime.isFinite() && deltaTime >= 0.0) { "Delta time must be finite and non-negative, got $deltaTime." }
}

private fun canonicalAxes(): List<Pair<Int, Int>> {
    return listOf(
        CanonicalKinematicState.PX to CanonicalKinematicState.VX,
        CanonicalKinematicState.PY to CanonicalKinematicState.VY,
        CanonicalKinematicState.PZ to CanonicalKinematicState.VZ,
    )
}

private fun canonicalTriples(): List<Triple<Int, Int, Int>> {
    return listOf(
        Triple(CanonicalKinematicState.PX, CanonicalKinematicState.VX, CanonicalKinematicState.AX),
        Triple(CanonicalKinematicState.PY, CanonicalKinematicState.VY, CanonicalKinematicState.AY),
        Triple(CanonicalKinematicState.PZ, CanonicalKinematicState.VZ, CanonicalKinematicState.AZ),
    )
}

private fun whiteJerkProcessNoise(deltaTime: Double, spectralDensity: Double): Matrix {
    var q = Matrix.zeros(CanonicalKinematicState.DIMENSION, CanonicalKinematicState.DIMENSION)
    val dt2 = deltaTime * deltaTime
    val dt3 = dt2 * deltaTime
    val dt4 = dt2 * dt2
    val dt5 = dt4 * deltaTime
    val dt6 = dt3 * dt3
    for ((p, v, a) in canonicalTriples()) {
        q = q.with(p, p, spectralDensity * dt6 / 36.0)
        q = q.with(p, v, spectralDensity * dt5 / 12.0)
        q = q.with(v, p, spectralDensity * dt5 / 12.0)
        q = q.with(p, a, spectralDensity * dt4 / 6.0)
        q = q.with(a, p, spectralDensity * dt4 / 6.0)
        q = q.with(v, v, spectralDensity * dt4 / 4.0)
        q = q.with(v, a, spectralDensity * dt3 / 2.0)
        q = q.with(a, v, spectralDensity * dt3 / 2.0)
        q = q.with(a, a, spectralDensity * dt2)
    }
    return q.symmetrized()
}
