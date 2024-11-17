from matrix import *


def lu_decomposition(coefficients: Matrix, results: Matrix | None = None) -> tuple[Matrix, Matrix]:
    # coefficients and results might be changed in outer code
    L = Matrix([[0] * coefficients.m for _ in range(coefficients.n)])
    U = Matrix([[value for value in row] for row in coefficients.values])

    # Straight Gaussian stroke
    for k in range(coefficients.n):
        if U[k][k] == 0:
            for i in range(k+1, coefficients.n):
                if U[i][k] != 0:
                    U[k], U[i] = U[i], U[k]
                    L[k], L[i] = L[i], L[k]
                    coefficients[k], coefficients[i] = coefficients[i], coefficients[k]
                    if results:
                        results[k], results[i] = results[i], results[k]
                    break
            else:
                print("There are not solutions")  # TODO: create custom Exception
                raise Exception
        L[k][k] = 1
        for i in range(k+1, coefficients.n):
            L[i][k] = U[i][k]/U[k][k]
            if U[i][k] == 0:
                continue
            for j in range(k, coefficients.m):
                U[i][j] -= L[i][k]*U[k][j]

    return L, U


def get_determinant(coefficients: Matrix) -> int | float:
    _, U = lu_decomposition(coefficients)
    det = 1
    for i in range(coefficients.n):
        det *= U[i][i]
    return det


def calculate_decisions(coefficients: Matrix, results: Matrix) -> Matrix:
    L, U = lu_decomposition(coefficients, results)
    res = results.copy()
    for k in range(res.m):
        for i in range(res.n):
            for j in range(i):
                res[i][k] -= res[j][k]*L[i][j]

    for k in range(res.m):
        for i in range(coefficients.n-1, -1, -1):
            for j in range(i+1, results.n):
                res[i][k] -= res[j][k]*U[i][j]
            res[i][k] /= U[i][i]
    return res


def get_inverse_matrix(matrix: Matrix) -> Matrix:
    E = Matrix([[1 if i == j else 0 for j in range(matrix.n)]for i in range(matrix.n)])
    return calculate_decisions(matrix, E)


if __name__ == "__main__":
    coefficient_matrix = Matrix([
        [-7, -9,  1, -9],
        [-6, -8, -5,  2],
        [-3,  6,  5, -9],
        [-2,  0, -5, -9]
    ])

    equation_roots = Matrix([
        [29],
        [42],
        [11],
        [75]
    ])

    print(calculate_decisions(coefficient_matrix, equation_roots))
    