# IMM Trajectory Solver

`imm-trajectory-solver` is a Kotlin/JVM backend library for 3D trajectory prediction with probabilistic multi-model filtering.

The public API supports two coordinate solvers:

- Cartesian solver: accepts Cartesian measurements and returns Cartesian predictions.
- WGS84 solver: accepts WGS84 geodetic measurements and returns WGS84 geodetic predictions.

Internally both solvers use the same metric IMM/Kalman core in canonical 9D state space. Coordinate conversion is isolated at the solver boundary:

```text
public coordinate input
-> CoordinateFrameAdapter
-> metric IMM core in canonical 9D meters
-> CoordinateFrameAdapter
-> public coordinate output
```

`trajectory-simulator` remains a separate project for benchmarks, scenarios, reports, and visualization. The backend library does not depend on simulator code.

## Project Scope

This project provides:

- local matrix and vector math primitives;
- typed domain objects for Cartesian and WGS84 measurements;
- typed trajectory prediction and update contracts;
- canonical 9D kinematic state representation;
- coordinate frame adapters for Cartesian and WGS84;
- linear and extended Kalman-family filter implementations;
- constant-velocity, constant-acceleration, and Singer/Gauss-Markov acceleration motion models;
- IMM state mixing, model probability updates, and combined prediction output;
- validation for dimensions, probabilities, covariance matrices, time ordering, frame compatibility, and gating.

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
    │   ├── coordinate/   # solver frames, WGS84 geodesy, coordinate adapters
    │   ├── domain/       # public measurements, predictions, updates, covariance types
    │   ├── filter/       # TrackingFilter, KF/EKF variants, update diagnostics
    │   ├── math/         # Matrix implementation and linear algebra helpers
    │   ├── measurement/  # internal metric observation models
    │   ├── motion/       # CV, CA, Singer, linear/nonlinear model contracts
    │   ├── solver/       # typed solver facades and metric IMM core
    │   ├── state/        # canonical 9D state index definitions
    │   └── validation/   # numerical validators and probability utilities
    └── test/kotlin/dev/trajectory/imm/
        ├── coordinate/
        ├── filter/
        ├── math/
        ├── solver/
        └── validation/
```

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

The public backend contract is typed by measurement and output position:

```kotlin
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
```

The solver uses explicit state passing:

- `initialize` creates a complete `SolverState`.
- `predictNext` does not mutate the solver state.
- `predictHorizon` uses temporary states and does not commit online state.
- `update` returns a `SolverStep<P>` containing the previous state, next state, and typed update diagnostics.
- callers advance online tracking by keeping `step.nextState`.

`SolverState` stores the active `SolverFrame`. Every `predictNext`, `predictHorizon`, and `update` call validates that the state belongs to the solver frame. This prevents accidental reuse of a Cartesian state in a WGS84 solver, or a WGS84 state from another origin.

## Solver Factory

Use `ImmTrajectorySolver.cartesian` for Cartesian coordinates:

```kotlin
val solver = ImmTrajectorySolver.cartesian()
```

Use `ImmTrajectorySolver.wgs84` for WGS84 coordinates. WGS84 requires an explicit origin at construction:

```kotlin
val origin = Wgs84Position(
    latitudeDegrees = 56.8389000,
    longitudeDegrees = 60.6057000,
    heightMeters = 250.0,
)

val solver = ImmTrajectorySolver.wgs84(origin = origin)
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

## Configurable IMM Filter Sets

Coordinate frame selection and IMM filter selection are independent:

- `ImmTrajectorySolver.cartesian(...)` or `ImmTrajectorySolver.wgs84(...)` chooses the public input/output coordinate system.
- `ImmFilterSetSpec` chooses the IMM motion models and Kalman filter implementations.

Available motion model types:

```kotlin
MotionModelType.CV
MotionModelType.CA
MotionModelType.SINGER
```

Available Kalman filter types:

```kotlin
KalmanFilterType.LINEAR
KalmanFilterType.EXTENDED
KalmanFilterType.INNOVATION_ADAPTIVE
KalmanFilterType.FADING_MEMORY
```

Example mixed IMM filter set:

```kotlin
val mixedFilterSet = ImmFilterSetSpec(
    models = listOf(
        ImmModelSpec("CV", MotionModelType.CV, KalmanFilterType.LINEAR),
        ImmModelSpec("CA", MotionModelType.CA, KalmanFilterType.INNOVATION_ADAPTIVE),
        ImmModelSpec("Singer", MotionModelType.SINGER, KalmanFilterType.EXTENDED),
    ),
)
```

The standard `CV`, `CA`, `Singer` names and order can use the default transition matrix even when filter implementations differ. Any non-standard model set must provide an explicit `transitionMatrix`.

Single-model example:

```kotlin
val cvOnly = ImmFilterSetSpec(
    models = listOf(
        ImmModelSpec("OnlyCV", MotionModelType.CV, KalmanFilterType.LINEAR),
    ),
    transitionMatrix = Matrix.ofRows(listOf(listOf(1.0))),
    initialProbabilities = mapOf("OnlyCV" to 1.0),
)
```

## Cartesian Usage Example

```kotlin
import dev.trajectory.imm.domain.CartesianPositionCovariance
import dev.trajectory.imm.domain.CartesianTimedMeasurement
import dev.trajectory.imm.domain.Vector3
import dev.trajectory.imm.solver.ImmFilterSetSpec
import dev.trajectory.imm.solver.ImmModelSpec
import dev.trajectory.imm.solver.ImmSolverConfig
import dev.trajectory.imm.solver.ImmTrajectorySolver
import dev.trajectory.imm.solver.KalmanFilterType
import dev.trajectory.imm.solver.MotionModelType

fun main() {
    val filterSet = ImmFilterSetSpec(
        models = listOf(
            ImmModelSpec("CV", MotionModelType.CV, KalmanFilterType.LINEAR),
            ImmModelSpec("CA", MotionModelType.CA, KalmanFilterType.INNOVATION_ADAPTIVE),
            ImmModelSpec("Singer", MotionModelType.SINGER, KalmanFilterType.EXTENDED),
        ),
    )

    val solver = ImmTrajectorySolver.cartesian(
        config = ImmSolverConfig.withIsotropicEnuMeasurementVariance(
            variance = 25.0,
        ),
        filterSet = filterSet,
    )

    val history = listOf(
        CartesianTimedMeasurement(time = 0.0, position = Vector3(0.0, 0.0, 1000.0)),
        CartesianTimedMeasurement(time = 1.0, position = Vector3(10.0, 1.0, 1002.0)),
        CartesianTimedMeasurement(time = 2.0, position = Vector3(20.0, 3.0, 1004.0)),
    )

    var state = solver.initialize(history)

    val prediction = solver.predictNext(
        state = state,
        toTime = 3.0,
    )

    val nextMeasurement = CartesianTimedMeasurement(
        time = 3.0,
        position = Vector3(31.0, 6.0, 1007.0),
    )

    val step = solver.update(
        state = state,
        measurement = nextMeasurement,
    )

    state = step.nextState

    val covariance = prediction.covariance as CartesianPositionCovariance
    println(prediction.expectedPosition)
    println(covariance.covariance.value)
    println(step.update.correctedPosition)
    println(step.update.modelProbabilities)
    println(state.time)
}
```

## WGS84 Usage Example

The WGS84 solver accepts and returns WGS84 positions. Internally it tracks in a local ENU tangent frame in meters relative to the explicit origin.

```kotlin
import dev.trajectory.imm.domain.TrajectoryPrediction
import dev.trajectory.imm.domain.Wgs84Position
import dev.trajectory.imm.domain.Wgs84PositionCovariance
import dev.trajectory.imm.domain.Wgs84TimedMeasurement
import dev.trajectory.imm.solver.ImmFilterSetSpec
import dev.trajectory.imm.solver.ImmModelSpec
import dev.trajectory.imm.solver.ImmSolverConfig
import dev.trajectory.imm.solver.ImmTrajectorySolver
import dev.trajectory.imm.solver.KalmanFilterType
import dev.trajectory.imm.solver.MotionModelType

fun main() {
    val origin = Wgs84Position(
        latitudeDegrees = 56.8389000,
        longitudeDegrees = 60.6057000,
        heightMeters = 250.0,
    )
    val filterSet = ImmFilterSetSpec(
        models = listOf(
            ImmModelSpec("CV", MotionModelType.CV, KalmanFilterType.LINEAR),
            ImmModelSpec("CA", MotionModelType.CA, KalmanFilterType.INNOVATION_ADAPTIVE),
            ImmModelSpec("Singer", MotionModelType.SINGER, KalmanFilterType.EXTENDED),
        ),
    )

    val solver = ImmTrajectorySolver.wgs84(
        origin = origin,
        config = ImmSolverConfig.withIsotropicEnuMeasurementVariance(
            variance = 25.0,
        ),
        filterSet = filterSet,
    )

    val history = listOf(
        Wgs84TimedMeasurement(
            time = 0.0,
            position = Wgs84Position(
                latitudeDegrees = 56.8389000,
                longitudeDegrees = 60.6057000,
                heightMeters = 250.0,
            ),
        ),
        Wgs84TimedMeasurement(
            time = 1.0,
            position = Wgs84Position(
                latitudeDegrees = 56.8389090,
                longitudeDegrees = 60.6057210,
                heightMeters = 254.0,
            ),
        ),
        Wgs84TimedMeasurement(
            time = 2.0,
            position = Wgs84Position(
                latitudeDegrees = 56.8389180,
                longitudeDegrees = 60.6057430,
                heightMeters = 258.5,
            ),
        ),
    )

    var state = solver.initialize(history)

    val prediction: TrajectoryPrediction<Wgs84Position> = solver.predictNext(
        state = state,
        toTime = 3.0,
    )

    val nextMeasurement = Wgs84TimedMeasurement(
        time = 3.0,
        position = Wgs84Position(
            latitudeDegrees = 56.8389270,
            longitudeDegrees = 60.6057650,
            heightMeters = 263.0,
        ),
    )

    val step = solver.update(
        state = state,
        measurement = nextMeasurement,
    )

    state = step.nextState

    val covariance = prediction.covariance as Wgs84PositionCovariance
    println(prediction.expectedPosition)
    println(covariance.enuCovariance.value)
    println(covariance.geodeticCovariance.value)
    println(step.update.correctedPosition)
    println(step.update.modelProbabilities)
    println(state.frame.id)
}
```

## Coordinate Frames

`CartesianImmTrajectorySolver` is a thin facade over the metric core. Its public coordinates and internal coordinates are the same metric Cartesian `Vector3`.

`Wgs84ImmTrajectorySolver` owns a `Wgs84CoordinateFrameAdapter`. The adapter performs:

- WGS84 geodetic to ECEF conversion;
- ECEF to local ENU conversion;
- local ENU to ECEF conversion;
- ECEF to WGS84 geodetic conversion;
- longitude normalization to `[-180, 180]`;
- measurement radius validation against the frame origin;
- covariance projection between ENU meters and `[latitudeDegrees, longitudeDegrees, heightMeters]^T`.

The WGS84 solver rejects:

- origins with `abs(latitudeDegrees) > 89.0`;
- predicted or measured positions near the poles;
- measurements outside `validRadiusMeters`;
- solver states from another frame.

## Measurement Noise and Covariance

Measurement noise is explicit:

```kotlin
sealed interface PositionMeasurementNoise

data class EnuPositionNoise(
    val covariance: CovarianceMatrix,
) : PositionMeasurementNoise

data class Wgs84PositionNoise(
    val covariance: CovarianceMatrix,
) : PositionMeasurementNoise
```

For Cartesian solvers, use `EnuPositionNoise`. The covariance is interpreted as metric Cartesian position covariance in meters squared.

For WGS84 solvers:

- `EnuPositionNoise` is passed directly to the metric core as local ENU covariance in meters squared.
- `Wgs84PositionNoise` is projected to local ENU at each measurement point.
- output covariance is `Wgs84PositionCovariance`.

`Wgs84PositionCovariance` contains:

- `enuCovariance`: internal metric local ENU covariance in meters squared;
- `geodeticCovariance`: covariance for `[latitudeDegrees, longitudeDegrees, heightMeters]^T`.

Geodetic covariance projection uses central finite differences with a 0.1 meter step in local ENU axes.

## WGS84 Assumptions

Current WGS84 support is intentionally local:

- datum: WGS84;
- height reference: ellipsoid height;
- tracking frame: local ENU tangent frame;
- origin: explicit `Wgs84Position` passed to `ImmTrajectorySolver.wgs84`;
- default valid radius: `100_000.0` meters;
- no automatic re-anchoring;
- no ECEF-native long-range IMM state in this refactor.

For long-range tracks, polar regions, or global paths crossing the validity radius, add a dedicated frame strategy before using the metric IMM core over large distances.

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
- Current implementation is constrained to canonical 9D state and 3D Cartesian metric measurements.
- Useful as the extension path for future nonlinear dynamics and radar-like measurement models.

`FadingMemoryKalmanFilter`

- Linear Kalman variant that inflates propagated covariance by a fading factor.
- Useful when old information should lose influence faster.

`InnovationAdaptiveKalmanFilter`

- Linear Kalman variant that adapts measurement noise from innovation magnitude.
- Useful for robustness against occasional measurement mismatch or changed measurement quality.

## Measurement Models

Measurement models are internal observation models in metric state space.

Cartesian position observation uses:

```text
z = [x, y, z]^T
```

`CartesianPositionMeasurementModel` observes position from the canonical 9D state with a 3x9 measurement matrix:

```kotlin
CartesianPositionMeasurementModel.isotropic(variance = 25.0)
```

There is no WGS84 measurement model in the Kalman layer. WGS84 conversion is handled only by `Wgs84CoordinateFrameAdapter`, before measurements enter the metric IMM core and after predictions leave it.

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
- compatible solver frame for every stateful operation;
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

The simulator wraps the Cartesian typed backend through its adapter module and converts simulator-side measurements, predictions, updates, and metadata to backend types. WGS84 benchmark scenarios can be added later on the simulator side without changing the backend architecture.

## Development Notes

- Keep public contracts explicit: interfaces, data classes, value classes, and immutable values.
- Keep IMM mixing in canonical 9D state space unless a future adapter defines mathematically valid conversion.
- Do not mix state estimates with different native dimensions directly.
- Keep coordinate conversion at solver boundaries, not inside Kalman measurement models.
- Keep `predictNext` and `predictHorizon` non-mutating.
- Treat out-of-sequence measurements as unsupported in the default update path.
- Add new benchmark logic to `trajectory-simulator`, not to this library.
