package dev.trajectory.imm.coordinate

import dev.trajectory.imm.domain.CartesianPositionCovariance
import dev.trajectory.imm.domain.CartesianTimedMeasurement
import dev.trajectory.imm.domain.CovarianceMatrix
import dev.trajectory.imm.domain.EnuPositionNoise
import dev.trajectory.imm.domain.PositionMeasurementNoise
import dev.trajectory.imm.domain.TimedMeasurement
import dev.trajectory.imm.domain.TrajectoryPrediction
import dev.trajectory.imm.domain.TrajectoryUpdate
import dev.trajectory.imm.domain.Vector3
import dev.trajectory.imm.domain.Wgs84Position
import dev.trajectory.imm.domain.Wgs84PositionCovariance
import dev.trajectory.imm.domain.Wgs84PositionNoise
import dev.trajectory.imm.domain.Wgs84TimedMeasurement
import dev.trajectory.imm.math.Matrix
import kotlin.math.abs
import kotlin.math.sqrt

interface CoordinateFrameAdapter<M : TimedMeasurement<P>, P> {
    val frame: SolverFrame

    fun toInternalMeasurement(measurement: M): CartesianTimedMeasurement

    fun fromInternalPrediction(prediction: TrajectoryPrediction<Vector3>): TrajectoryPrediction<P>

    fun fromInternalUpdate(update: TrajectoryUpdate<Vector3>): TrajectoryUpdate<P>
}

class CartesianCoordinateFrameAdapter(
    private val measurementNoise: EnuPositionNoise,
) : CoordinateFrameAdapter<CartesianTimedMeasurement, Vector3> {
    override val frame: SolverFrame = CartesianSolverFrame

    override fun toInternalMeasurement(measurement: CartesianTimedMeasurement): CartesianTimedMeasurement {
        return measurement.copy(
            measurementNoiseOverride = measurement.measurementNoiseOverride ?: measurementNoise.covariance,
        )
    }

    override fun fromInternalPrediction(prediction: TrajectoryPrediction<Vector3>): TrajectoryPrediction<Vector3> {
        return prediction.copy(
            covariance = CartesianPositionCovariance((prediction.covariance as CartesianPositionCovariance).covariance),
        )
    }

    override fun fromInternalUpdate(update: TrajectoryUpdate<Vector3>): TrajectoryUpdate<Vector3> {
        return update.copy(
            prediction = fromInternalPrediction(update.prediction),
        )
    }
}

class Wgs84CoordinateFrameAdapter(
    override val frame: Wgs84SolverFrame,
    private val measurementNoise: PositionMeasurementNoise,
) : CoordinateFrameAdapter<Wgs84TimedMeasurement, Wgs84Position> {
    override fun toInternalMeasurement(measurement: Wgs84TimedMeasurement): CartesianTimedMeasurement {
        val enu = Wgs84Geodesy.geodeticToLocalEnu(
            position = measurement.position,
            origin = frame.origin,
        )
        requireWithinValidRadius(enu, "WGS84 measurement")
        return CartesianTimedMeasurement(
            time = measurement.time,
            position = enu,
            measurementNoiseOverride = projectedMeasurementNoise(measurement.position),
        )
    }

    override fun fromInternalPrediction(prediction: TrajectoryPrediction<Vector3>): TrajectoryPrediction<Wgs84Position> {
        val enuCovariance = (prediction.covariance as CartesianPositionCovariance).covariance
        val expectedPosition = toWgs84(prediction.expectedPosition, "WGS84 prediction")
        return TrajectoryPrediction(
            time = prediction.time,
            expectedPosition = expectedPosition,
            covariance = Wgs84PositionCovariance(
                enuCovariance = enuCovariance,
                geodeticCovariance = projectEnuCovarianceToGeodetic(prediction.expectedPosition, enuCovariance),
            ),
            internalStateEstimate = prediction.internalStateEstimate,
            modelProbabilities = prediction.modelProbabilities,
            metadata = prediction.metadata,
        )
    }

    override fun fromInternalUpdate(update: TrajectoryUpdate<Vector3>): TrajectoryUpdate<Wgs84Position> {
        val correctedPosition = toWgs84(update.correctedPosition, "WGS84 corrected position")
        return TrajectoryUpdate(
            time = update.time,
            prediction = fromInternalPrediction(update.prediction),
            correctedPosition = correctedPosition,
            internalUpdatedEstimate = update.internalUpdatedEstimate,
            accepted = update.accepted,
            logLikelihood = update.logLikelihood,
            mahalanobisDistanceSquared = update.mahalanobisDistanceSquared,
            modelProbabilities = update.modelProbabilities,
            metadata = update.metadata,
        )
    }

    fun measurementNoiseAt(position: Wgs84Position): CovarianceMatrix {
        return projectedMeasurementNoise(position)
    }

    private fun projectedMeasurementNoise(position: Wgs84Position): CovarianceMatrix {
        return when (measurementNoise) {
            is EnuPositionNoise -> measurementNoise.covariance
            is Wgs84PositionNoise -> {
                val jacobian = wgs84ToEnuJacobian(position)
                CovarianceMatrix((jacobian * measurementNoise.covariance.value * jacobian.transpose()).symmetrized())
            }
        }
    }

    private fun toWgs84(
        enu: Vector3,
        label: String,
    ): Wgs84Position {
        requireWithinValidRadius(enu, label)
        val position = Wgs84Geodesy.localEnuToGeodetic(enu, frame.origin)
        require(abs(position.latitudeDegrees) <= MAX_ABS_ORIGIN_LATITUDE_DEGREES) {
            "$label latitude is in near-pole zone: ${position.latitudeDegrees}."
        }
        return position
    }

    private fun projectEnuCovarianceToGeodetic(
        enuPosition: Vector3,
        enuCovariance: CovarianceMatrix,
    ): CovarianceMatrix {
        val jacobian = enuToWgs84Jacobian(enuPosition)
        return CovarianceMatrix((jacobian * enuCovariance.value * jacobian.transpose()).symmetrized())
    }

    private fun enuToWgs84Jacobian(enuPosition: Vector3): Matrix {
        val columns = listOf(
            Vector3(FINITE_DIFFERENCE_STEP_METERS, 0.0, 0.0),
            Vector3(0.0, FINITE_DIFFERENCE_STEP_METERS, 0.0),
            Vector3(0.0, 0.0, FINITE_DIFFERENCE_STEP_METERS),
        ).map { step ->
            val plus = Wgs84Geodesy.localEnuToGeodetic(enuPosition + step, frame.origin)
            val minus = Wgs84Geodesy.localEnuToGeodetic(enuPosition - step, frame.origin)
            listOf(
                (plus.latitudeDegrees - minus.latitudeDegrees) / (2.0 * FINITE_DIFFERENCE_STEP_METERS),
                Wgs84Geodesy.longitudeDeltaDegrees(plus.longitudeDegrees, minus.longitudeDegrees) / (2.0 * FINITE_DIFFERENCE_STEP_METERS),
                (plus.heightMeters - minus.heightMeters) / (2.0 * FINITE_DIFFERENCE_STEP_METERS),
            )
        }
        return Matrix.ofRows(
            listOf(
                listOf(columns[0][0], columns[1][0], columns[2][0]),
                listOf(columns[0][1], columns[1][1], columns[2][1]),
                listOf(columns[0][2], columns[1][2], columns[2][2]),
            ),
        )
    }

    private fun wgs84ToEnuJacobian(position: Wgs84Position): Matrix {
        val latitudeStep = 1.0e-6
        val longitudeStep = 1.0e-6
        val heightStep = FINITE_DIFFERENCE_STEP_METERS
        val columns = listOf(
            Perturbation(
                plus = position.copy(latitudeDegrees = (position.latitudeDegrees + latitudeStep).coerceAtMost(90.0)),
                minus = position.copy(latitudeDegrees = (position.latitudeDegrees - latitudeStep).coerceAtLeast(-90.0)),
                denominator = 2.0 * latitudeStep,
            ),
            Perturbation(
                plus = position.copy(longitudeDegrees = Wgs84Geodesy.normalizeLongitudeDegrees(position.longitudeDegrees + longitudeStep)),
                minus = position.copy(longitudeDegrees = Wgs84Geodesy.normalizeLongitudeDegrees(position.longitudeDegrees - longitudeStep)),
                denominator = 2.0 * longitudeStep,
            ),
            Perturbation(
                plus = position.copy(heightMeters = position.heightMeters + heightStep),
                minus = position.copy(heightMeters = position.heightMeters - heightStep),
                denominator = 2.0 * heightStep,
            ),
        ).map { perturbation ->
            val plus = Wgs84Geodesy.geodeticToLocalEnu(perturbation.plus, frame.origin)
            val minus = Wgs84Geodesy.geodeticToLocalEnu(perturbation.minus, frame.origin)
            listOf(
                (plus.x - minus.x) / perturbation.denominator,
                (plus.y - minus.y) / perturbation.denominator,
                (plus.z - minus.z) / perturbation.denominator,
            )
        }
        return Matrix.ofRows(
            listOf(
                listOf(columns[0][0], columns[1][0], columns[2][0]),
                listOf(columns[0][1], columns[1][1], columns[2][1]),
                listOf(columns[0][2], columns[1][2], columns[2][2]),
            ),
        )
    }

    private fun requireWithinValidRadius(
        enu: Vector3,
        label: String,
    ) {
        val distance = sqrt(enu.x * enu.x + enu.y * enu.y + enu.z * enu.z)
        require(distance <= frame.validRadiusMeters) {
            "$label distance $distance m exceeds WGS84 local frame valid radius ${frame.validRadiusMeters} m."
        }
    }

    private data class Perturbation(
        val plus: Wgs84Position,
        val minus: Wgs84Position,
        val denominator: Double,
    )

    companion object {
        const val FINITE_DIFFERENCE_STEP_METERS: Double = 0.1
    }
}

private operator fun Vector3.plus(other: Vector3): Vector3 {
    return Vector3(x + other.x, y + other.y, z + other.z)
}

private operator fun Vector3.minus(other: Vector3): Vector3 {
    return Vector3(x - other.x, y - other.y, z - other.z)
}
