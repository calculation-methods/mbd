# 9 вариант
from typing import Callable


def rectangular_method(f: Callable, x_start: float, x_end: float, step: float) -> float:
    x = x_start
    res = 0
    while x < x_end:
        res += f((2*x + step)/2)
        x += step
    return step*res


def trapeze_method(f: Callable, x_start: float, x_end: float, step: float) -> float:
    x = x_start+step
    res = f(x_start)/2 + f(x_end)/2
    while x < x_end:
        res += f(x)
        x += step
    return step * res


def simpson_method(f: Callable, x_start: float, x_end: float, step: float) -> float:
    x = x_start + step
    res = f(x_start) + f(x_end)
    flag = True
    while x < x_end:
        res += f(x) * (4 if flag else 2)
        x += step
        flag = not flag
    return step * res / 3


def RRR_estimation(F_1: float, F_2: float, step_1: float, step_2: float, p: float) -> float:
    return F_1 + (F_1 - F_2)/((step_2/step_1)**p - 1)


if __name__ == "__main__":
    y = lambda x: x/(x**2 + 9)
    x_0, x_k, precision = 0, 2, (0.5, 0.25, 0.000001)

    # y = lambda x: x/((3*x + 4)**2)
    # x_0, x_k, precision = -1, 1, (0.5, 0.25, 0.000001)

    methods = [
        {"name": "Rectangular", "function": rectangular_method},
        {"name": "Trapeze", "function": trapeze_method},
        {"name": "Simpson", "function": simpson_method}
    ]

    for method in methods:
        print(f'{method["name"]} method')
        F = []
        for h in precision:
            F.append(method["function"](y, x_0, x_k, h))
            print(f'\tIntegral = {F[-1]}\t\t(Step = {h})')
        print(f'\tIntegral = {RRR_estimation(F[0], F[1], precision[0], precision[1], 2)} \t\t(Runge-Romberg-Richardson estimation, step 1 = {precision[0]}, step 2 = {precision[1]})')
        print('\n')
