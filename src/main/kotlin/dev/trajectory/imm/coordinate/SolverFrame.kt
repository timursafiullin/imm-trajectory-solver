package dev.trajectory.imm.coordinate

import dev.trajectory.imm.domain.Wgs84Position
import kotlin.math.abs

sealed interface SolverFrame {
    val id: String
}

data object CartesianSolverFrame : SolverFrame {
    override val id: String = "cartesian"
}

enum class GeodeticDatum {
    WGS84,
}

enum class HeightReference {
    ELLIPSOID,
}

data class Wgs84FrameConfig(
    val origin: Wgs84Position,
    val datum: GeodeticDatum = GeodeticDatum.WGS84,
    val heightReference: HeightReference = HeightReference.ELLIPSOID,
    val validRadiusMeters: Double = 100_000.0,
    val id: String = buildWgs84FrameId(origin, datum, heightReference, validRadiusMeters),
) {
    init {
        require(abs(origin.latitudeDegrees) <= MAX_ABS_ORIGIN_LATITUDE_DEGREES) {
            "WGS84 local ENU origin latitude must be outside near-pole zone: |latitude| <= $MAX_ABS_ORIGIN_LATITUDE_DEGREES."
        }
        require(validRadiusMeters.isFinite() && validRadiusMeters > 0.0) {
            "WGS84 valid radius must be finite and positive."
        }
        require(id.isNotBlank()) { "WGS84 frame id must not be blank." }
    }
}

data class Wgs84SolverFrame(
    override val id: String,
    val origin: Wgs84Position,
    val datum: GeodeticDatum = GeodeticDatum.WGS84,
    val heightReference: HeightReference = HeightReference.ELLIPSOID,
    val validRadiusMeters: Double = 100_000.0,
) : SolverFrame {
    init {
        require(abs(origin.latitudeDegrees) <= MAX_ABS_ORIGIN_LATITUDE_DEGREES) {
            "WGS84 local ENU origin latitude must be outside near-pole zone: |latitude| <= $MAX_ABS_ORIGIN_LATITUDE_DEGREES."
        }
        require(validRadiusMeters.isFinite() && validRadiusMeters > 0.0) {
            "WGS84 valid radius must be finite and positive."
        }
        require(id.isNotBlank()) { "WGS84 frame id must not be blank." }
    }

    companion object {
        fun fromConfig(config: Wgs84FrameConfig): Wgs84SolverFrame {
            return Wgs84SolverFrame(
                id = config.id,
                origin = config.origin,
                datum = config.datum,
                heightReference = config.heightReference,
                validRadiusMeters = config.validRadiusMeters,
            )
        }
    }
}

const val MAX_ABS_ORIGIN_LATITUDE_DEGREES: Double = 89.0

fun buildWgs84FrameId(
    origin: Wgs84Position,
    datum: GeodeticDatum,
    heightReference: HeightReference,
    validRadiusMeters: Double,
): String {
    return "wgs84:${datum.name}:${heightReference.name}:${origin.latitudeDegrees}:${origin.longitudeDegrees}:${origin.heightMeters}:$validRadiusMeters"
}
