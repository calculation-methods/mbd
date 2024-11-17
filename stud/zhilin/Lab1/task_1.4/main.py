import math

from matrix import *
from matrix import Vector, Matrix


def get_indexes(matrix1: Matrix) -> tuple[int, int]:
    """
    return indexes of maximum non-diagonal element
    """
    m: float = matrix1[0][1]
    indexes: tuple[int, int] = (0, 1)
    for i in range(matrix1.n):
        for j in range(i+1, matrix1.m):
            if abs(matrix1[i][j]) > m:
                m = abs(matrix1[i][j])
                indexes = (i, j)
    return indexes


def get_phi(a_ij: float, a_ii: float, a_jj: float) -> float:
    if a_ii == a_jj:
        return math.pi/4
    return math.atan(2*a_ij/(a_ii-a_jj)) / 2


def get_U_matrix(matrix1: Matrix) -> Matrix:
    res = [[1 if i == j else 0 for j in range(matrix1.n)] for i in range(matrix1.n)]
    i_max, j_max = get_indexes(matrix1)
    phi = get_phi(a_ij=matrix1[i_max][j_max], a_ii=matrix1[i_max][i_max], a_jj=matrix1[j_max][j_max])
    res[i_max][i_max] = math.cos(phi)
    res[j_max][j_max] = math.cos(phi)
    res[i_max][j_max] = -math.sin(phi)
    res[j_max][i_max] = math.sin(phi)
    return Matrix(res)


def get_eps(matrix1: Matrix) -> float:
    eps = 0
    for i in range(matrix1.n):
        for j in range(i+1, matrix1.m):
            eps += matrix1[i][j]**2
    return math.sqrt(eps)


def Jacobi_method(coeff_matrix: Matrix, EPS: float) -> tuple[Vector, Matrix]:
    eigenvectors = Matrix([[1 if i == j else 0 for j in range(coeff_matrix.n)]for i in range(coeff_matrix.n)])
    while get_eps(coeff_matrix) > EPS:
        U = get_U_matrix(coeff_matrix)
        eigenvectors = multiple_matrix(eigenvectors, U)
        coeff_matrix = multiple_matrix(multiple_matrix(U.transposed(), coeff_matrix), U)
    eigenvalues = Vector([coeff_matrix[i][i] for i in range(coeff_matrix.n)])
    return eigenvalues, eigenvectors


if __name__ == "__main__":
    coefficient_matrix = Matrix([
        [4, 7, -1],
        [7, -9, -6],
        [-1, -6, -4]
    ])

    values, vectors = Jacobi_method(coefficient_matrix, EPS=0.01)
    print("Собственные значения:", *[f'λ_{i+1} = {values[i]}' for i in range(values.n)], sep='\n', end='\n\n')

    print("Собственные векторы:")
    for i in range(vectors.n):
        print(f'x_{i+1} = {[vectors[j][i] for j in range(vectors.n)]}')
