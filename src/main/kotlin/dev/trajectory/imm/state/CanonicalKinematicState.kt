package dev.trajectory.imm.state

object CanonicalKinematicState {
    const val DIMENSION: Int = 9
    const val PX: Int = 0
    const val PY: Int = 1
    const val PZ: Int = 2
    const val VX: Int = 3
    const val VY: Int = 4
    const val VZ: Int = 5
    const val AX: Int = 6
    const val AY: Int = 7
    const val AZ: Int = 8

    val POSITION_INDICES: IntArray = intArrayOf(PX, PY, PZ)
    val VELOCITY_INDICES: IntArray = intArrayOf(VX, VY, VZ)
    val ACCELERATION_INDICES: IntArray = intArrayOf(AX, AY, AZ)
}
