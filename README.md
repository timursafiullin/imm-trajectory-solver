# Interacting Multi-Model Trajectory Solver

Backend Kotlin/JVM library for 3D trajectory prediction with probabilistic multi-model filtering.

This project contains:

- linear algebra primitives;
- domain types for measurements, estimates, and predictions;
- motion and measurement models;
- Kalman-family filters;
- IMM model mixing and probability updates.

Benchmark scenarios, simulator pipelines, reports, and visualization belong to the separate `trajectory-simulator` project.

## Requirements

- JDK 17 or newer.
- Gradle Wrapper from this repository.

## Build

```powershell
.\gradlew.bat build
```
