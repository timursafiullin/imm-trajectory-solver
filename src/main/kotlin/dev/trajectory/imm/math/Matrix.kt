package dev.trajectory.imm.math

import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

class Matrix private constructor(
    val rows: Int,
    val columns: Int,
    private val data: DoubleArray,
) {
    init {
        require(rows > 0) { "Matrix row count must be positive." }
        require(columns > 0) { "Matrix column count must be positive." }
        require(data.size == rows * columns) {
            "Matrix data size ${data.size} does not match dimensions ${rows}x$columns."
        }
        require(data.all { it.isFinite() }) { "Matrix values must be finite." }
    }

    operator fun get(row: Int, column: Int): Double {
        require(row in 0 until rows) { "Matrix row index $row is outside 0 until $rows." }
        require(column in 0 until columns) { "Matrix column index $column is outside 0 until $columns." }
        return data[index(row, column)]
    }

    fun with(row: Int, column: Int, value: Double): Matrix {
        require(value.isFinite()) { "Matrix value must be finite." }
        require(row in 0 until rows) { "Matrix row index $row is outside 0 until $rows." }
        require(column in 0 until columns) { "Matrix column index $column is outside 0 until $columns." }
        val copy = data.copyOf()
        copy[index(row, column)] = value
        return Matrix(rows, columns, copy)
    }

    fun toDoubleArray(): DoubleArray = data.copyOf()

    fun toNestedLists(): List<List<Double>> {
        return List(rows) { row ->
            List(columns) { column -> this[row, column] }
        }
    }

    operator fun plus(other: Matrix): Matrix {
        requireSameShape(other, "add")
        return Matrix(rows, columns, DoubleArray(data.size) { index -> data[index] + other.data[index] })
    }

    operator fun minus(other: Matrix): Matrix {
        requireSameShape(other, "subtract")
        return Matrix(rows, columns, DoubleArray(data.size) { index -> data[index] - other.data[index] })
    }

    operator fun unaryMinus(): Matrix {
        return Matrix(rows, columns, DoubleArray(data.size) { index -> -data[index] })
    }

    operator fun times(scalar: Double): Matrix {
        require(scalar.isFinite()) { "Matrix scalar multiplier must be finite." }
        return Matrix(rows, columns, DoubleArray(data.size) { index -> data[index] * scalar })
    }

    operator fun div(scalar: Double): Matrix {
        require(scalar.isFinite() && abs(scalar) > 0.0) { "Matrix scalar divisor must be finite and non-zero." }
        return this * (1.0 / scalar)
    }

    operator fun times(other: Matrix): Matrix {
        require(columns == other.rows) {
            "Cannot multiply ${rows}x$columns matrix by ${other.rows}x${other.columns} matrix."
        }
        val result = DoubleArray(rows * other.columns)
        for (row in 0 until rows) {
            for (column in 0 until other.columns) {
                var sum = 0.0
                for (k in 0 until columns) {
                    sum += this[row, k] * other[k, column]
                }
                result[row * other.columns + column] = sum
            }
        }
        return Matrix(rows, other.columns, result)
    }

    fun transpose(): Matrix {
        val result = DoubleArray(rows * columns)
        for (row in 0 until rows) {
            for (column in 0 until columns) {
                result[column * rows + row] = this[row, column]
            }
        }
        return Matrix(columns, rows, result)
    }

    fun trace(): Double {
        requireSquare("compute trace")
        var sum = 0.0
        for (i in 0 until rows) {
            sum += this[i, i]
        }
        return sum
    }

    fun symmetrized(): Matrix {
        requireSquare("symmetrize")
        return (this + transpose()) * 0.5
    }

    fun isSymmetric(tolerance: Double = DEFAULT_TOLERANCE): Boolean {
        require(tolerance >= 0.0) { "Tolerance must be non-negative." }
        if (rows != columns) {
            return false
        }
        for (row in 0 until rows) {
            for (column in row + 1 until columns) {
                if (abs(this[row, column] - this[column, row]) > tolerance) {
                    return false
                }
            }
        }
        return true
    }

    fun isPositiveSemiDefinite(tolerance: Double = DEFAULT_TOLERANCE): Boolean {
        require(tolerance >= 0.0) { "Tolerance must be non-negative." }
        if (!isSymmetric(max(tolerance, DEFAULT_TOLERANCE))) {
            return false
        }
        val n = rows
        val l = Array(n) { DoubleArray(n) }
        for (i in 0 until n) {
            for (j in 0..i) {
                var sum = this[i, j]
                for (k in 0 until j) {
                    sum -= l[i][k] * l[j][k]
                }
                if (i == j) {
                    if (sum < -tolerance) {
                        return false
                    }
                    l[i][j] = if (sum <= tolerance) 0.0 else sqrt(sum)
                } else {
                    l[i][j] = if (abs(l[j][j]) <= tolerance) {
                        if (abs(sum) > tolerance * 10.0) {
                            return false
                        }
                        0.0
                    } else {
                        sum / l[j][j]
                    }
                }
            }
        }
        return true
    }

    fun solve(rightHandSide: Matrix): Matrix {
        requireSquare("solve a linear system")
        require(rightHandSide.rows == rows) {
            "Right-hand side row count ${rightHandSide.rows} must equal coefficient size $rows."
        }
        val n = rows
        val rhsColumns = rightHandSide.columns
        val a = Array(n) { row -> DoubleArray(n) { column -> this[row, column] } }
        val b = Array(n) { row -> DoubleArray(rhsColumns) { column -> rightHandSide[row, column] } }

        for (pivotIndex in 0 until n) {
            var pivotRow = pivotIndex
            var pivotAbs = abs(a[pivotIndex][pivotIndex])
            for (candidate in pivotIndex + 1 until n) {
                val candidateAbs = abs(a[candidate][pivotIndex])
                if (candidateAbs > pivotAbs) {
                    pivotRow = candidate
                    pivotAbs = candidateAbs
                }
            }
            require(pivotAbs > SINGULAR_TOLERANCE) {
                "Matrix is singular or ill-conditioned at pivot $pivotIndex."
            }
            if (pivotRow != pivotIndex) {
                val tmpA = a[pivotIndex]
                a[pivotIndex] = a[pivotRow]
                a[pivotRow] = tmpA
                val tmpB = b[pivotIndex]
                b[pivotIndex] = b[pivotRow]
                b[pivotRow] = tmpB
            }

            val pivot = a[pivotIndex][pivotIndex]
            for (row in pivotIndex + 1 until n) {
                val factor = a[row][pivotIndex] / pivot
                a[row][pivotIndex] = 0.0
                for (column in pivotIndex + 1 until n) {
                    a[row][column] -= factor * a[pivotIndex][column]
                }
                for (column in 0 until rhsColumns) {
                    b[row][column] -= factor * b[pivotIndex][column]
                }
            }
        }

        val x = Array(n) { DoubleArray(rhsColumns) }
        for (rhsColumn in 0 until rhsColumns) {
            for (row in n - 1 downTo 0) {
                var sum = b[row][rhsColumn]
                for (column in row + 1 until n) {
                    sum -= a[row][column] * x[column][rhsColumn]
                }
                x[row][rhsColumn] = sum / a[row][row]
            }
        }
        return ofRows(x.map { it.toList() })
    }

    fun inverse(): Matrix {
        requireSquare("invert")
        return solve(identity(rows))
    }

    fun determinant(): Double {
        requireSquare("compute determinant")
        val decomposition = gaussianEliminationForDeterminant()
        return decomposition.sign * decomposition.diagonalProduct
    }

    fun logDeterminantPositive(): Double {
        requireSquare("compute log determinant")
        val decomposition = gaussianEliminationForDeterminant()
        val determinantSign = decomposition.sign * decomposition.diagonalProduct.sign()
        require(determinantSign > 0.0) {
            "Matrix determinant must be positive for log determinant."
        }
        return decomposition.logAbsDiagonalProduct
    }

    private fun gaussianEliminationForDeterminant(): DeterminantDecomposition {
        val n = rows
        val a = Array(n) { row -> DoubleArray(n) { column -> this[row, column] } }
        var sign = 1.0
        var diagonalProduct = 1.0
        var logAbsDiagonalProduct = 0.0
        for (pivotIndex in 0 until n) {
            var pivotRow = pivotIndex
            var pivotAbs = abs(a[pivotIndex][pivotIndex])
            for (candidate in pivotIndex + 1 until n) {
                val candidateAbs = abs(a[candidate][pivotIndex])
                if (candidateAbs > pivotAbs) {
                    pivotRow = candidate
                    pivotAbs = candidateAbs
                }
            }
            if (pivotAbs <= SINGULAR_TOLERANCE) {
                return DeterminantDecomposition(sign = 0.0, diagonalProduct = 0.0, logAbsDiagonalProduct = Double.NEGATIVE_INFINITY)
            }
            if (pivotRow != pivotIndex) {
                val tmp = a[pivotIndex]
                a[pivotIndex] = a[pivotRow]
                a[pivotRow] = tmp
                sign *= -1.0
            }
            val pivot = a[pivotIndex][pivotIndex]
            diagonalProduct *= pivot
            logAbsDiagonalProduct += ln(abs(pivot))
            for (row in pivotIndex + 1 until n) {
                val factor = a[row][pivotIndex] / pivot
                a[row][pivotIndex] = 0.0
                for (column in pivotIndex + 1 until n) {
                    a[row][column] -= factor * a[pivotIndex][column]
                }
            }
        }
        return DeterminantDecomposition(sign = sign, diagonalProduct = diagonalProduct, logAbsDiagonalProduct = logAbsDiagonalProduct)
    }

    private fun requireSameShape(other: Matrix, operation: String) {
        require(rows == other.rows && columns == other.columns) {
            "Cannot $operation matrices with shapes ${rows}x$columns and ${other.rows}x${other.columns}."
        }
    }

    private fun requireSquare(operation: String) {
        require(rows == columns) { "Cannot $operation a non-square ${rows}x$columns matrix." }
    }

    private fun index(row: Int, column: Int): Int = row * columns + column

    override fun equals(other: Any?): Boolean {
        return other is Matrix &&
            rows == other.rows &&
            columns == other.columns &&
            data.contentEquals(other.data)
    }

    override fun hashCode(): Int {
        var result = rows
        result = 31 * result + columns
        result = 31 * result + data.contentHashCode()
        return result
    }

    override fun toString(): String = "Matrix(rows=$rows, columns=$columns, data=${toNestedLists()})"

    private data class DeterminantDecomposition(
        val sign: Double,
        val diagonalProduct: Double,
        val logAbsDiagonalProduct: Double,
    )

    companion object {
        const val DEFAULT_TOLERANCE: Double = 1.0e-9
        private const val SINGULAR_TOLERANCE: Double = 1.0e-12

        fun zeros(rows: Int, columns: Int): Matrix {
            return Matrix(rows, columns, DoubleArray(rows * columns))
        }

        fun identity(size: Int): Matrix {
            require(size > 0) { "Identity matrix size must be positive." }
            val values = DoubleArray(size * size)
            for (i in 0 until size) {
                values[i * size + i] = 1.0
            }
            return Matrix(size, size, values)
        }

        fun diagonal(values: List<Double>): Matrix {
            require(values.isNotEmpty()) { "Diagonal matrix must contain at least one value." }
            require(values.all { it.isFinite() }) { "Diagonal values must be finite." }
            val size = values.size
            val data = DoubleArray(size * size)
            for (i in values.indices) {
                data[i * size + i] = values[i]
            }
            return Matrix(size, size, data)
        }

        fun columnVector(values: List<Double>): Matrix {
            require(values.isNotEmpty()) { "Column vector must contain at least one value." }
            require(values.all { it.isFinite() }) { "Column vector values must be finite." }
            return Matrix(values.size, 1, values.toDoubleArray())
        }

        fun ofRows(rows: List<List<Double>>): Matrix {
            require(rows.isNotEmpty()) { "Matrix must contain at least one row." }
            val columnCount = rows.first().size
            require(columnCount > 0) { "Matrix must contain at least one column." }
            require(rows.all { it.size == columnCount }) { "All matrix rows must have the same column count." }
            val data = DoubleArray(rows.size * columnCount)
            for (row in rows.indices) {
                for (column in 0 until columnCount) {
                    val value = rows[row][column]
                    require(value.isFinite()) { "Matrix values must be finite." }
                    data[row * columnCount + column] = value
                }
            }
            return Matrix(rows.size, columnCount, data)
        }

        fun outerProduct(leftColumnVector: Matrix, rightColumnVector: Matrix): Matrix {
            require(leftColumnVector.columns == 1) {
                "Left outer-product argument must be a column vector, got ${leftColumnVector.rows}x${leftColumnVector.columns}."
            }
            require(rightColumnVector.columns == 1) {
                "Right outer-product argument must be a column vector, got ${rightColumnVector.rows}x${rightColumnVector.columns}."
            }
            return leftColumnVector * rightColumnVector.transpose()
        }
    }
}

operator fun Double.times(matrix: Matrix): Matrix = matrix * this

private fun Double.sign(): Double {
    return when {
        this > 0.0 -> 1.0
        this < 0.0 -> -1.0
        else -> 0.0
    }
}
