package dev.trajectory.imm.solver

import dev.trajectory.imm.domain.Prediction
import dev.trajectory.imm.domain.TimedMeasurement
import dev.trajectory.imm.domain.UpdateResult
import dev.trajectory.imm.filter.FilterState

interface TrajectoryPredictionBackend {
    fun initialize(measurements: List<TimedMeasurement>): SolverState

    fun predictNext(
        state: SolverState,
        toTime: Double,
    ): Prediction

    fun update(
        state: SolverState,
        measurement: TimedMeasurement,
    ): SolverStep

    fun predictHorizon(
        state: SolverState,
        futureTimes: List<Double>,
    ): List<Prediction>

    fun reset(): SolverState
}

data class SolverState(
    val time: Double,
    val filterStates: Map<String, FilterState>,
    val modelProbabilities: Map<String, Double>,
)

data class SolverStep(
    val previousState: SolverState,
    val nextState: SolverState,
    val update: UpdateResult,
)
