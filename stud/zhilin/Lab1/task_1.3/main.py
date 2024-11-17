from matrix import *


def preprocessing(coefficients: Matrix, results: Matrix) -> None:
    n, m = coefficients.get_shape()
    for i in range(n):
        a = coefficients[i][i]
        if a == 0:
            continue
        results[i][0] /= a
        for j in range(m):
            if i == j:
                coefficients[i][j] = 0
            else:
                coefficients[i][j] /= -a


def get_current_eps(vect1: Matrix, vect2: Matrix) -> float:
    eps = 0
    for i in range(vect1.get_shape()[0]):
        eps += (vect1[i][0] - vect2[i][0]) ** 2
    return eps ** 0.5


def simple_iterations(a: Matrix, b: Matrix, EPS0: float) -> tuple[int, Matrix]:
    x_previous = Matrix(b.values)
    k = 0
    while get_current_eps(x_current := plus_matrix(b, multiple_matrix(a, x_previous)), x_previous) > EPS0:
        k += 1
        x_current, x_previous = None, x_current
    return k, x_previous


def seidel_method(a: Matrix, b: Matrix, EPS0: float) -> tuple[int, Matrix]:
    x_previous = Matrix(b.values)
    x_current = x_previous.copy()
    flag = True
    k, eps = 0, 0
    while flag or eps > EPS0:
        k += 1
        flag = False
        for i in range(b.get_shape()[0]):
            x_current[i][0] = 0
            for j in range(b.get_shape()[0]):
                if i == j:
                    x_current[i][0] += b[i][0]
                elif i < j:
                    x_current[i][0] += a[i][j]*x_previous[j][0]
                else:
                    x_current[i][0] += a[i][j]*x_current[j][0]
        eps = get_current_eps(x_current, x_previous)
        x_current, x_previous = x_current, x_current.copy()
    return k, x_previous


if __name__ == "__main__":
    coefficient_matrix = Matrix([
        [12, -3, -1, 3],
        [5, 20, 9, 1],
        [6, -3, -21, -7],
        [8, -7, 3, -27]
    ])

    equation_roots = Matrix([
        [-31],
        [90],
        [119],
        [71]
    ])

    preprocessing(coefficient_matrix, equation_roots)

    last_iteration, res_matrix = simple_iterations(coefficient_matrix, equation_roots, 0.001)
    print(last_iteration)
    print(res_matrix)

    print('\n\n')

    last_iteration, res_matrix = seidel_method(coefficient_matrix, equation_roots, 0.001)
    print(last_iteration)
    print(res_matrix)
