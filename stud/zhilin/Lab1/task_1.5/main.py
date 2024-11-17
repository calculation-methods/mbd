from matrix import *
from matrix import Vector, Matrix


coefficient_matrix: Matrix


def get_eps(matrix1: Matrix) -> float:
    eps = 0
    for i in range(matrix1.n):
        for j in range(i-1):
            eps += matrix1[i][j]**2
    return eps**0.5


def sign(a: float) -> int:
    return 1 if a >= 0 else -1


def get_H_matrix(coefficients: Matrix, ind: int) -> Matrix:
    E = Matrix([[1 if i == j else 0 for j in range(coefficients.n)] for i in range(coefficients.n)])
    v = []
    for i in range(coefficients.n):
        if i < ind:
            v.append([0])
        elif i == ind:
            v.append([coefficients[i][i] + sign(coefficients[i][i]) * (sum([coefficients[j][i]**2 for j in range(ind, coefficients.n)])**0.5)])
        else:
            v.append([coefficients[i][ind]])
    v = Matrix(v)
    k = -multiple_matrix(v.transposed(), v)[0][0]/2
    V = multiple_matrix(v, v.transposed())
    for i in range(V.n):
        for j in range(V.n):
            V[i][j] /= k
    H = plus_matrix(E, V)
    return H


def QR_decomposition(coefficients: Matrix) -> tuple[Matrix, Matrix]:
    coefficients = coefficients.copy()
    Q = get_H_matrix(coefficients, 0)
    coefficients = multiple_matrix(Q, coefficients)
    for i in range(1, coefficients.n-1):
        H = get_H_matrix(coefficients, i)
        Q = multiple_matrix(Q, get_H_matrix(coefficients, i))
        coefficients = multiple_matrix(H, coefficients)
    return Q, coefficients


def get_eigenvalues(coefficients: Matrix, EPS: float) -> list[float]:
    while get_eps(coefficients) > EPS:
        Q, R = QR_decomposition(coefficients)
        coefficients = multiple_matrix(R, Q)
    return [coefficients[i][i] for i in range(coefficients.n)]


if __name__ == "__main__":
    coefficient_matrix = Matrix([
        [-5, -8, 4],
        [4, 2, 6],
        [-2, 5, -6]
    ])

    eigenvalues = get_eigenvalues(coefficient_matrix, 0.01)

    print(*[f'λ_{i+1} = {eigenvalues[i]}' for i in range(eigenvalues.__len__())], sep='\n')
