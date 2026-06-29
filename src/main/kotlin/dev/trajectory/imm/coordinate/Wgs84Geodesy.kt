package dev.trajectory.imm.coordinate

import dev.trajectory.imm.domain.Vector3
import dev.trajectory.imm.domain.Wgs84Position
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

object Wgs84Geodesy {
    const val SEMI_MAJOR_AXIS_METERS: Double = 6_378_137.0
    const val FLATTENING: Double = 1.0 / 298.257_223_563
    const val FIRST_ECCENTRICITY_SQUARED: Double = FLATTENING * (2.0 - FLATTENING)

    private const val ECEF_TO_GEODETIC_TOLERANCE: Double = 1.0e-12
    private const val ECEF_TO_GEODETIC_MAX_ITERATIONS: Int = 12

    fun geodeticToEcef(position: Wgs84Position): Vector3 {
        val latitudeRadians = Math.toRadians(position.latitudeDegrees)
        val longitudeRadians = Math.toRadians(position.longitudeDegrees)
        val sinLatitude = sin(latitudeRadians)
        val cosLatitude = cos(latitudeRadians)
        val sinLongitude = sin(longitudeRadians)
        val cosLongitude = cos(longitudeRadians)
        val primeVerticalRadius = SEMI_MAJOR_AXIS_METERS / sqrt(1.0 - FIRST_ECCENTRICITY_SQUARED * sinLatitude * sinLatitude)

        return Vector3(
            x = (primeVerticalRadius + position.heightMeters) * cosLatitude * cosLongitude,
            y = (primeVerticalRadius + position.heightMeters) * cosLatitude * sinLongitude,
            z = (primeVerticalRadius * (1.0 - FIRST_ECCENTRICITY_SQUARED) + position.heightMeters) * sinLatitude,
        )
    }

    fun ecefToGeodetic(ecef: Vector3): Wgs84Position {
        val longitudeRadians = atan2(ecef.y, ecef.x)
        val horizontalRadius = sqrt(ecef.x * ecef.x + ecef.y * ecef.y)
        var latitudeRadians = atan2(ecef.z, horizontalRadius * (1.0 - FIRST_ECCENTRICITY_SQUARED))
        var heightMeters = 0.0

        repeat(ECEF_TO_GEODETIC_MAX_ITERATIONS) {
            val sinLatitude = sin(latitudeRadians)
            val primeVerticalRadius = SEMI_MAJOR_AXIS_METERS / sqrt(1.0 - FIRST_ECCENTRICITY_SQUARED * sinLatitude * sinLatitude)
            heightMeters = horizontalRadius / cos(latitudeRadians) - primeVerticalRadius
            val nextLatitude = atan2(
                ecef.z,
                horizontalRadius * (1.0 - FIRST_ECCENTRICITY_SQUARED * primeVerticalRadius / (primeVerticalRadius + heightMeters)),
            )
            if (abs(nextLatitude - latitudeRadians) <= ECEF_TO_GEODETIC_TOLERANCE) {
                latitudeRadians = nextLatitude
                return Wgs84Position(
                    latitudeDegrees = Math.toDegrees(latitudeRadians),
                    longitudeDegrees = normalizeLongitudeDegrees(Math.toDegrees(longitudeRadians)),
                    heightMeters = heightMeters,
                )
            }
            latitudeRadians = nextLatitude
        }

        val sinLatitude = sin(latitudeRadians)
        val primeVerticalRadius = SEMI_MAJOR_AXIS_METERS / sqrt(1.0 - FIRST_ECCENTRICITY_SQUARED * sinLatitude * sinLatitude)
        heightMeters = horizontalRadius / cos(latitudeRadians) - primeVerticalRadius
        return Wgs84Position(
            latitudeDegrees = Math.toDegrees(latitudeRadians),
            longitudeDegrees = normalizeLongitudeDegrees(Math.toDegrees(longitudeRadians)),
            heightMeters = heightMeters,
        )
    }

    fun ecefToLocalEnu(
        ecef: Vector3,
        origin: Wgs84Position,
    ): Vector3 {
        val originEcef = geodeticToEcef(origin)
        val dx = ecef.x - originEcef.x
        val dy = ecef.y - originEcef.y
        val dz = ecef.z - originEcef.z
        val latitudeRadians = Math.toRadians(origin.latitudeDegrees)
        val longitudeRadians = Math.toRadians(origin.longitudeDegrees)
        val sinLatitude = sin(latitudeRadians)
        val cosLatitude = cos(latitudeRadians)
        val sinLongitude = sin(longitudeRadians)
        val cosLongitude = cos(longitudeRadians)

        return Vector3(
            x = -sinLongitude * dx + cosLongitude * dy,
            y = -sinLatitude * cosLongitude * dx - sinLatitude * sinLongitude * dy + cosLatitude * dz,
            z = cosLatitude * cosLongitude * dx + cosLatitude * sinLongitude * dy + sinLatitude * dz,
        )
    }

    fun localEnuToEcef(
        enu: Vector3,
        origin: Wgs84Position,
    ): Vector3 {
        val originEcef = geodeticToEcef(origin)
        val latitudeRadians = Math.toRadians(origin.latitudeDegrees)
        val longitudeRadians = Math.toRadians(origin.longitudeDegrees)
        val sinLatitude = sin(latitudeRadians)
        val cosLatitude = cos(latitudeRadians)
        val sinLongitude = sin(longitudeRadians)
        val cosLongitude = cos(longitudeRadians)
        val dx = -sinLongitude * enu.x - sinLatitude * cosLongitude * enu.y + cosLatitude * cosLongitude * enu.z
        val dy = cosLongitude * enu.x - sinLatitude * sinLongitude * enu.y + cosLatitude * sinLongitude * enu.z
        val dz = cosLatitude * enu.y + sinLatitude * enu.z

        return Vector3(
            x = originEcef.x + dx,
            y = originEcef.y + dy,
            z = originEcef.z + dz,
        )
    }

    fun geodeticToLocalEnu(
        position: Wgs84Position,
        origin: Wgs84Position,
    ): Vector3 {
        return ecefToLocalEnu(
            ecef = geodeticToEcef(position),
            origin = origin,
        )
    }

    fun localEnuToGeodetic(
        enu: Vector3,
        origin: Wgs84Position,
    ): Wgs84Position {
        return ecefToGeodetic(localEnuToEcef(enu, origin))
    }

    fun normalizeLongitudeDegrees(longitudeDegrees: Double): Double {
        require(longitudeDegrees.isFinite()) { "Longitude must be finite." }
        var normalized = (longitudeDegrees + 180.0) % 360.0
        if (normalized < 0.0) {
            normalized += 360.0
        }
        normalized -= 180.0
        return if (normalized == -180.0 && longitudeDegrees > 0.0) 180.0 else normalized
    }

    fun longitudeDeltaDegrees(
        leftDegrees: Double,
        rightDegrees: Double,
    ): Double {
        var delta = normalizeLongitudeDegrees(leftDegrees) - normalizeLongitudeDegrees(rightDegrees)
        if (delta > 180.0) delta -= 360.0
        if (delta < -180.0) delta += 360.0
        require(delta.isFinite() && abs(delta) <= 180.0) { "Longitude delta must be finite." }
        return delta
    }
}
