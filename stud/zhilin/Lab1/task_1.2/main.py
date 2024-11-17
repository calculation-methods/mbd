from matrix import *


def tridiagonal_algorithm(coefficients: Matrix, results: Matrix) -> Matrix:
    if coefficients.n < 3 or coefficients.m < 3:
        print("Incorrect shapes of matrices")
        raise Exception
    a, b, c, d = 0, coefficients[0][0], coefficients[0][1], results.values[0][0]
    P, Q = [0]*coefficients.m, [0]*coefficients.m

    P[0], Q[0] = -c/b, d/b
    for i in range(1, coefficients.n-1):
        a, b, c, d = coefficients[i][i-1], coefficients[i][i], coefficients[i][i+1], results[i][0]
        P[i] = -c/(b + a*P[i-1])
        Q[i] = (d - a*Q[i-1])/(b + a*P[i-1])
    a, b, c, d = coefficients[-1][-2], coefficients[-1][-1], 0, results[-1][0]
    Q[-1] = (d - a * Q[-2]) / (b + a * P[-2])

    decisions = [0]*results.n
    decisions[-1] = Q[-1]
    for i in range(decisions.__len__()-2, -1, -1):
        decisions[i] = P[i]*decisions[i+1] + Q[i]

    return Matrix([[i] for i in decisions])


if __name__ == "__main__":
    coefficient_matrix = Matrix([
        [8, -4, 0, 0, 0],
        [-2, 12, -7, 0, 0],
        [0, 2, -9, 1, 0],
        [0, 0, -8, 17, -4],
        [0, 0, 0, -7, 13]
    ])

    equation_roots = Matrix([
        [32],
        [15],
        [-10],
        [133],
        [-76]
    ])

    print(tridiagonal_algorithm(coefficient_matrix, equation_roots))
