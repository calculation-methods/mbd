from typing import Callable
import math

f1 = lambda x: x[0]**2 + x[1]**2 - 16
f2 = lambda x: x[0] - math.e**x[1] + 4


def newton_method(x_0: list[float], EPS: float) -> tuple[list[float], int]:
    """
    Функция f1(x1, x2): x1^2 + x2^2 - 16
    Функция f2(x1, x2): x1 - e^(x2) + 4

    Первая производная f1 по x1: 2*x1
    Первая производная f1 по x2: 2*x2

    Первая производная f2 по x1: 1
    Первая производная f2 по x2: -e^(x2)

    Начальные значения (подбираются по графикам):
    x1 = -3.5
    x2 = -2
    """

    def eps(vect1: list[float], vect2: list[float]) -> float:
        return max((abs(vect1[i]-vect2[i]) for i in range(len(vect1))))

    def det(x: list[float], matrix: list[list[Callable]]) -> float:
        return matrix[0][0](x) * matrix[1][1](x) - matrix[0][1](x) * matrix[1][0](x)

    df1_x1 = lambda x: 2*x[0]
    df1_x2 = lambda x: 2*x[1]
    df2_x1 = lambda x: 1
    df2_x2 = lambda x: -math.e**x[1]

    A1 = [
        [f1, df1_x2],
        [f2, df2_x2]
    ]

    A2 = [
        [df1_x1, f1],
        [df2_x1, f2]
    ]

    J = [
        [df1_x1, df1_x2],
        [df2_x1, df2_x2]
    ]

    A = [A1, A2]

    x_next, x_curr, k = [x_0[i] - det(x_0, A[i])/det(x=x_0, matrix=J) for i in range(len(x_0))], x_0, 1
    while eps(x_curr, x_next) >= EPS:
        k += 1
        x_next, x_curr = [x_next[i] - det(x_next, A[i])/det(x=x_next, matrix=J) for i in range(len(x_next))], x_next
    return x_next, k


def simple_iterations_method(x_0: list[float], q: float, EPS: float) -> tuple[list[float], int]:
    """
    Функция f1(x1, x2): x1^2 + x2^2 - 16
    Функция f2(x1, x2): x1 - e^(x2) + 4

    Уравнение с выделенным членом (x1 = phi1(x1, x2)): x1 = e^(x2) - 4
    Уравнение с выделенным членом (x2 = phi2(x1, x2)): x2 = sqrt(16 - x1^2)

    phi1(x1, x2) = e^(x2) - 4
    phi1_dx1(x1, x2) = 0
    phi1_dx2(x1, x2) = e^(x2)

    phi2(x1, x2) = sqrt(16 - x1^2)
    phi1_dx1(x1, x2) = -x1/sqrt(16 - x1^2)
    phi1_dx2(x1, x2) = 0
    """

    def eps(vect1: list[float], vect2: list[float]) -> float:
        return q*max((abs(vect1[i] - vect2[i]) for i in range(len(vect1))))/(1-q)

    phi1 = lambda x: math.e ** x[1] - 4
    phi2 = lambda x: -math.sqrt(16-x[0]**2)

    x_next, x_curr, k = x_0, [3*i for i in x_0], 0
    while eps(x_curr, x_next) >= EPS:
        k += 1
        x_next, x_curr = [phi1(x_next), phi2(x_next)], x_next
    return x_next, k


if __name__ == "__main__":
    deviation = 1e-9

    result, number_of_iterations = newton_method(x_0=[-3.5, -2], EPS=deviation)
    print('\nМетод Ньютона')
    print(f'Результат (x_k): {result}\nКоличество итераций (k): {number_of_iterations}')
    print(f'Значение функций при данном корне x_k: {f1(result), f2(result)}')
    print()

    result, number_of_iterations = simple_iterations_method(x_0=[-3.5, -2], q=0.9, EPS=deviation)
    print('\nМетод простых итераций')
    print(f'Результат (x_k): {result}\nКоличество итераций (k): {number_of_iterations}')
    print(f'Значение функций при данном корне x_k: {f1(result), f2(result)}')
    print()
