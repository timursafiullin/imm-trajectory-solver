# IMM Trajectory Solver

`imm-trajectory-solver` is a Kotlin/JVM backend library for 3D trajectory prediction with probabilistic multi-model filtering.

The library estimates the hidden kinematic state of one moving object from time-stamped Cartesian position measurements and predicts future positions using an Interacting Multiple Model (IMM) filter. It owns only numerical filtering, motion models, measurement models, validation, and backend prediction contracts.

Benchmark scenarios, synthetic trajectories, metrics, reports, CSV/JSON output, and visualization belong to the separate `trajectory-simulator` project.

## Project Scope

This project provides:

- local matrix and vector math primitives;
- domain types for measurements, state estimates, predictions, and update diagnostics;
- canonical 9D kinematic state representation;
- Cartesian position measurement model;
- linear and extended Kalman-family filter implementations;
- constant-velocity, constant-acceleration, and Singer/Gauss-Markov acceleration motion models;
- IMM state mixing, model probability updates, and combined prediction output;
- validation for dimensions, probabilities, covariance matrices, time ordering, and gating.

This project does not provide:

- benchmark pipelines;
- simulator scenarios;
- charting, animation, or report generation;
- CSV/JSON experiment IO;
- application/demo entry points;
- simulator-side solver protocol.

## Project Layout

```text
imm-trajectory-solver/
├── build.gradle.kts
├── settings.gradle.kts
└── src/
    ├── main/kotlin/dev/trajectory/imm/
    │   ├── domain/       # public measurement, estimate, prediction, and update types
    │   ├── filter/       # TrackingFilter, KF/EKF variants, update diagnostics
    │   ├── math/         # Matrix implementation and linear algebra helpers
    │   ├── measurement/  # Cartesian measurement model and measurement abstractions
    │   ├── motion/       # CV, CA, Singer, linear/nonlinear model contracts
    │   ├── solver/       # TrajectoryPredictionBackend and ImmTrajectorySolver
    │   ├── state/        # canonical 9D state index definitions
    │   └── validation/   # numerical validators and probability utilities
    └── test/kotlin/dev/trajectory/imm/
        ├── filter/
        ├── math/
        ├── solver/
        └── validation/
```

`imm-trajectory-solver` is currently a single Gradle Kotlin/JVM library project. It is intentionally independent from `trajectory-simulator`.

## Requirements

- JDK 17+
- Gradle Wrapper
- Kotlin/JVM

The project uses Gradle Kotlin DSL, Kotlin `2.3.0`, JUnit 5 through `kotlin("test")`, and JVM bytecode target 17.

If your default Java is older than 17, set `JAVA_HOME` before running Gradle. Example for PowerShell:

```powershell
$env:JAVA_HOME='C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot'
$env:PATH="$env:JAVA_HOME\bin;$env:PATH"
```

## Commands

Run all tests:

```powershell
.\gradlew.bat test
```

Build the library:

```powershell
.\gradlew.bat build
```

Clean build outputs:

```powershell
.\gradlew.bat clean
```

## Core State Model

The current backend uses one canonical IMM state space for all built-in models:

```text
[px, py, pz, vx, vy, vz, ax, ay, az]^T
```

Dimension: `9`

Index constants are defined in:

```text
dev.trajectory.imm.state.CanonicalKinematicState
```

The canonical state is required because IMM mixing is only valid when all member estimates live in the same state space. Built-in `CV`, `CA`, and `Singer` filters all expose this canonical 9D state directly.

## Public Backend API

The main backend contract is:

```kotlin
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
```

The solver uses explicit state passing:

- `initialize` creates a complete `SolverState`.
- `predictNext` does not mutate the solver state.
- `predictHorizon` uses temporary states and does not commit online state.
- `update` returns a `SolverStep` containing the previous state, next state, and diagnostics.
- callers advance online tracking by keeping `step.nextState`.

## Default IMM Solver

The easiest entry point is:

```kotlin
val solver = ImmTrajectorySolver.defaultCartesian()
```

The default solver contains three canonical 9D linear Kalman filters:

- `CV` - constant velocity with acceleration compatibility slots;
- `CA` - constant acceleration with white-jerk process noise;
- `Singer` - Gauss-Markov acceleration model with maneuver-time memory.

Default model transition matrix:

```text
        to CV  to CA  to Singer
CV      0.95   0.04   0.01
CA      0.03   0.94   0.03
Singer  0.02   0.08   0.90
```

Default initial model probabilities are equal:

```text
CV = 1/3, CA = 1/3, Singer = 1/3
```

## Usage Example

```kotlin
import dev.trajectory.imm.domain.TimedMeasurement
import dev.trajectory.imm.domain.Vector3
import dev.trajectory.imm.solver.ImmTrajectorySolver

fun main() {
    val solver = ImmTrajectorySolver.defaultCartesian(
        measurementVariance = 25.0,
    )

    val history = listOf(
        TimedMeasurement(time = 0.0, position = Vector3(0.0, 0.0, 1000.0)),
        TimedMeasurement(time = 1.0, position = Vector3(10.0, 1.0, 1002.0)),
        TimedMeasurement(time = 2.0, position = Vector3(20.0, 3.0, 1004.0)),
    )

    var state = solver.initialize(history)

    val prediction = solver.predictNext(
        state = state,
        toTime = 3.0,
    )

    val nextMeasurement = TimedMeasurement(
        time = 3.0,
        position = Vector3(31.0, 6.0, 1007.0),
    )

    val step = solver.update(
        state = state,
        measurement = nextMeasurement,
    )

    state = step.nextState

    println(prediction.expectedMeasurement.mean.value)
    println(step.update.modelProbabilities)
    println(state.time)
}
```

## Available Motion Models

`ConstantVelocityModel9D`

- Canonical 9D state.
- Position is propagated by velocity.
- Acceleration slots are kept for IMM compatibility.
- Useful for near-linear motion and short-horizon prediction.

`ConstantAccelerationModel9D`

- Canonical 9D state.
- Position, velocity, and acceleration are propagated with constant-acceleration dynamics.
- Process noise is based on white jerk.
- Useful for acceleration and braking segments.

`SingerAccelerationModel9D`

- Canonical 9D state.
- Acceleration follows a Gauss-Markov decay process.
- Configurable maneuver time and acceleration noise intensity.
- Useful for maneuvering targets where acceleration has memory but should decay over time.

## Available Filters

`LinearKalmanFilter`

- Uses a linear motion model and a linear measurement model.
- Applies Joseph-form covariance update.
- Computes Gaussian innovation log-likelihood and Mahalanobis distance.
- Supports gating by Mahalanobis distance.

`ExtendedKalmanFilter`

- Uses a nonlinear motion model and an EKF measurement model.
- Current implementation is constrained to canonical 9D state and 3D Cartesian measurements.
- Useful as the extension path for future nonlinear dynamics and radar-like measurement models.

`FadingMemoryKalmanFilter`

- Linear Kalman variant that inflates propagated covariance by a fading factor.
- Useful when old information should lose influence faster.

`InnovationAdaptiveKalmanFilter`

- Linear Kalman variant that adapts measurement noise from innovation magnitude.
- Useful for robustness against occasional measurement mismatch or changed measurement quality.

## Measurement Model

The implemented measurement model is:

```text
z = [x, y, z]^T
```

`CartesianPositionMeasurementModel` observes position from the canonical 9D state with a 3x9 measurement matrix. The helper constructor:

```kotlin
CartesianPositionMeasurementModel.isotropic(variance = 25.0)
```

creates isotropic Cartesian measurement noise.

Future radar or spherical measurements require nonlinear observation logic and EKF/UKF/CKF support. They should not be forced into the current linear Cartesian measurement contract.

## Numerical Behavior

The backend validates:

- non-empty filter set;
- unique filter names;
- canonical 9D state dimension for built-in IMM members;
- square transition matrix with one row per filter;
- transition matrix rows normalized to one;
- finite, positive gating threshold;
- strictly increasing initialization and update times;
- normalized model probabilities;
- covariance symmetry and positive semi-definiteness within tolerance;
- measurement and state vector dimensions.

Kalman-family updates use:

- innovation covariance validation;
- Mahalanobis-distance gating;
- Gaussian log-likelihood;
- Joseph-form covariance update for accepted corrections;
- covariance symmetrization after propagation and mixture operations.

IMM model probabilities are updated using log-likelihoods and log-sum-exp normalization.

## Integration With `trajectory-simulator`

The dependency direction is one-way:

```text
trajectory-simulator may depend on imm-trajectory-solver.
imm-trajectory-solver must not depend on trajectory-simulator.
```

The simulator wraps this backend through its adapter module and converts simulator-side measurements, predictions, updates, and metadata to backend types. This library should not import simulator packages or implement simulator-specific reporting, visualization, or benchmark concerns.

## Development Notes

- Keep public contracts explicit: interfaces, data classes, value classes, and immutable values.
- Keep IMM mixing in canonical 9D state space unless a future adapter defines mathematically valid conversion.
- Do not mix state estimates with different native dimensions directly.
- Keep `predictNext` and `predictHorizon` non-mutating.
- Treat out-of-sequence measurements as unsupported in the default update path.
- Add new benchmark logic to `trajectory-simulator`, not to this library.
