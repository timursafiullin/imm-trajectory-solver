package dev.trajectory.imm.math

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class MatrixTest {
    @Test
    fun `matrix arithmetic and multiplication preserve expected values`() {
        val a = Matrix.ofRows(
            listOf(
                listOf(1.0, 2.0),
                listOf(3.0, 4.0),
            ),
        )
        val b = Matrix.ofRows(
            listOf(
                listOf(5.0, 6.0),
                listOf(7.0, 8.0),
            ),
        )

        val sum = a + b
        assertEquals(6.0, sum[0, 0], 1.0e-12)
        assertEquals(12.0, sum[1, 1], 1.0e-12)

        val product = a * b
        assertEquals(19.0, product[0, 0], 1.0e-12)
        assertEquals(22.0, product[0, 1], 1.0e-12)
        assertEquals(43.0, product[1, 0], 1.0e-12)
        assertEquals(50.0, product[1, 1], 1.0e-12)
    }

    @Test
    fun `inverse and solve recover identity and right hand side`() {
        val matrix = Matrix.ofRows(
            listOf(
                listOf(4.0, 7.0),
                listOf(2.0, 6.0),
            ),
        )
        val inverse = matrix.inverse()
        val identity = matrix * inverse
        assertEquals(1.0, identity[0, 0], 1.0e-10)
        assertEquals(0.0, identity[0, 1], 1.0e-10)
        assertEquals(0.0, identity[1, 0], 1.0e-10)
        assertEquals(1.0, identity[1, 1], 1.0e-10)

        val rhs = Matrix.columnVector(listOf(1.0, 0.0))
        val solution = matrix.solve(rhs)
        val recovered = matrix * solution
        assertEquals(1.0, recovered[0, 0], 1.0e-10)
        assertEquals(0.0, recovered[1, 0], 1.0e-10)
    }

    @Test
    fun `determinant log determinant and symmetrization work`() {
        val matrix = Matrix.ofRows(
            listOf(
                listOf(2.0, 0.5),
                listOf(0.5, 3.0),
            ),
        )
        assertEquals(5.75, matrix.determinant(), 1.0e-12)
        assertEquals(kotlin.math.ln(5.75), matrix.logDeterminantPositive(), 1.0e-12)
        assertTrue(matrix.isPositiveSemiDefinite())

        val nonSymmetric = Matrix.ofRows(
            listOf(
                listOf(1.0, 2.0),
                listOf(4.0, 1.0),
            ),
        )
        val symmetric = nonSymmetric.symmetrized()
        assertEquals(3.0, symmetric[0, 1], 1.0e-12)
        assertEquals(3.0, symmetric[1, 0], 1.0e-12)
    }

    @Test
    fun `dimension mismatches fail loudly`() {
        val left = Matrix.zeros(2, 3)
        val right = Matrix.zeros(2, 2)

        assertFailsWith<IllegalArgumentException> {
            left + right
        }
        assertFailsWith<IllegalArgumentException> {
            left * right
        }
    }

    private fun assertEquals(expected: Double, actual: Double, tolerance: Double) {
        assertTrue(abs(expected - actual) <= tolerance, "Expected $expected, got $actual.")
    }
}
