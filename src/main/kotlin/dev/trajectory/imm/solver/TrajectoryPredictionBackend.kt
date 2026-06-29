package dev.trajectory.imm.solver

import dev.trajectory.imm.coordinate.CartesianSolverFrame
import dev.trajectory.imm.coordinate.SolverFrame
import dev.trajectory.imm.domain.TimedMeasurement
import dev.trajectory.imm.domain.TrajectoryPrediction
import dev.trajectory.imm.domain.TrajectoryUpdate
import dev.trajectory.imm.filter.FilterState

interface TrajectoryPredictionBackend<M : TimedMeasurement<P>, P> {
    fun initialize(measurements: List<M>): SolverState

    fun predictNext(
        state: SolverState,
        toTime: Double,
    ): TrajectoryPrediction<P>

    fun update(
        state: SolverState,
        measurement: M,
    ): SolverStep<P>

    fun predictHorizon(
        state: SolverState,
        futureTimes: List<Double>,
    ): List<TrajectoryPrediction<P>>

    fun reset(): SolverState
}

data class SolverState(
    val time: Double,
    val filterStates: Map<String, FilterState>,
    val modelProbabilities: Map<String, Double>,
    val frame: SolverFrame = CartesianSolverFrame,
)

data class SolverStep<P>(
    val previousState: SolverState,
    val nextState: SolverState,
    val update: TrajectoryUpdate<P>,
)
