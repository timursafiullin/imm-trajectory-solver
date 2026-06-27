package dev.trajectory.imm.validation

import dev.trajectory.imm.math.Matrix
import kotlin.test.Test
import kotlin.test.assertFailsWith

class ValidatorsTest {
    @Test
    fun `transition matrix rows must be stochastic`() {
        Validators.requireTransitionMatrix(
            Matrix.ofRows(
                listOf(
                    listOf(0.9, 0.1),
                    listOf(0.2, 0.8),
                ),
            ),
            expectedSize = 2,
        )

        assertFailsWith<IllegalArgumentException> {
            Validators.requireTransitionMatrix(
                Matrix.ofRows(
                    listOf(
                        listOf(0.9, 0.3),
                        listOf(0.2, 0.8),
                    ),
                ),
                expectedSize = 2,
            )
        }
    }

    @Test
    fun `probabilities must be normalized over expected model names`() {
        Validators.requireProbabilities(
            probabilities = mapOf("CV" to 0.5, "CA" to 0.5),
            expectedNames = listOf("CV", "CA"),
        )

        assertFailsWith<IllegalArgumentException> {
            Validators.requireProbabilities(
                probabilities = mapOf("CV" to 0.7, "CA" to 0.5),
                expectedNames = listOf("CV", "CA"),
            )
        }
    }

    @Test
    fun `covariance must be symmetric positive semi definite`() {
        Validators.requireCovariance(
            Matrix.ofRows(
                listOf(
                    listOf(2.0, 0.25),
                    listOf(0.25, 1.0),
                ),
            ),
            name = "valid",
        )

        assertFailsWith<IllegalArgumentException> {
            Validators.requireCovariance(
                Matrix.ofRows(
                    listOf(
                        listOf(1.0, 2.0),
                        listOf(2.0, -1.0),
                    ),
                ),
                name = "invalid",
            )
        }
    }
}
