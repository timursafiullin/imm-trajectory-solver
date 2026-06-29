package dev.trajectory.imm.coordinate

import dev.trajectory.imm.domain.Wgs84Position
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class Wgs84GeodesyTest {
    private val origin = Wgs84Position(
        latitudeDegrees = 56.8389,
        longitudeDegrees = 60.6057,
        heightMeters = 250.0,
    )

    @Test
    fun `origin converts to zero local ENU`() {
        val enu = Wgs84Geodesy.geodeticToLocalEnu(
            position = origin,
            origin = origin,
        )

        assertEquals(0.0, enu.x, 1.0e-9)
        assertEquals(0.0, enu.y, 1.0e-9)
        assertEquals(0.0, enu.z, 1.0e-9)
    }

    @Test
    fun `longitude increase produces positive east displacement`() {
        val enu = Wgs84Geodesy.geodeticToLocalEnu(
            position = origin.copy(longitudeDegrees = origin.longitudeDegrees + 0.001),
            origin = origin,
        )

        assertTrue(enu.x > 0.0, "Expected positive east displacement, got ${enu.x}.")
        assertTrue(abs(enu.y) < enu.x * 0.01, "Expected north cross-coupling to stay small, got ${enu.y}.")
    }

    @Test
    fun `latitude increase produces positive north displacement`() {
        val enu = Wgs84Geodesy.geodeticToLocalEnu(
            position = origin.copy(latitudeDegrees = origin.latitudeDegrees + 0.001),
            origin = origin,
        )

        assertTrue(enu.y > 0.0, "Expected positive north displacement, got ${enu.y}.")
        assertTrue(abs(enu.x) < enu.y * 0.01, "Expected east cross-coupling to stay small, got ${enu.x}.")
    }

    @Test
    fun `height increase produces positive up displacement`() {
        val enu = Wgs84Geodesy.geodeticToLocalEnu(
            position = origin.copy(heightMeters = origin.heightMeters + 100.0),
            origin = origin,
        )

        assertEquals(0.0, enu.x, 1.0e-9)
        assertEquals(0.0, enu.y, 1.0e-9)
        assertEquals(100.0, enu.z, 1.0e-8)
    }

    @Test
    fun `local ENU round trip preserves WGS84 position`() {
        val position = Wgs84Position(
            latitudeDegrees = 56.8392,
            longitudeDegrees = 60.6064,
            heightMeters = 320.0,
        )
        val enu = Wgs84Geodesy.geodeticToLocalEnu(position, origin)
        val roundTrip = Wgs84Geodesy.localEnuToGeodetic(enu, origin)

        assertEquals(position.latitudeDegrees, roundTrip.latitudeDegrees, 1.0e-9)
        assertEquals(position.longitudeDegrees, roundTrip.longitudeDegrees, 1.0e-9)
        assertEquals(position.heightMeters, roundTrip.heightMeters, 1.0e-5)
    }

    @Test
    fun `longitude normalization keeps antimeridian values in range`() {
        assertEquals(-179.9, Wgs84Geodesy.normalizeLongitudeDegrees(180.1), 1.0e-12)
        assertEquals(179.9, Wgs84Geodesy.normalizeLongitudeDegrees(-180.1), 1.0e-12)
    }

    @Test
    fun `WGS84 frame rejects near-pole origin`() {
        assertFailsWith<IllegalArgumentException> {
            Wgs84FrameConfig(
                origin = Wgs84Position(
                    latitudeDegrees = 89.5,
                    longitudeDegrees = 10.0,
                    heightMeters = 0.0,
                ),
            )
        }
    }
}
