import math


def f(x: float) -> float:
    return math.acos(x)


def lagrange_interpolation(x: float, cords: list[tuple[float, float]]) -> float:
    coefficients, n = [cords[i][1] for i in range(len(cords))], len(cords)
    for i in range(n):
        for j in range(n):
            if i != j:
                coefficients[i] /= cords[i][0]-cords[j][0]
    for i in range(n):
        for j in range(n):
            if i != j:
                coefficients[i] *= x-cords[j][0]
    return sum(coefficients)


def newton_interpolation(x: float, cords: list[tuple[float, float]]) -> float:
    coefficients, n = [cords[i][1] for i in range(len(cords))], len(cords)
    for i in range(1, n):
        for j in range(n-1, i-1, -1):
            coefficients[j] = (coefficients[j]-coefficients[j-1])/(cords[j][0]-cords[j-i][0])
            
    for i in range(1, n):
        for j in range(i):
            coefficients[i] *= x-cords[j][0]
    return sum(coefficients)


if __name__ == '__main__':
    x_vect = [-0.4, -0.1, 0.2, 0.5]
    x_marked = 0.1
    cord = [(x, f(x)) for x in x_vect]

    print('\nМногочлен Лагранжа')
    print(f'\tРезультат: {lagrange_interpolation(x_marked, cord)}')
    print(f'\tОшибка: {abs(lagrange_interpolation(x_marked, cord) - f(x_marked))}')
    print()

    print('Многочлен Ньютона')
    print(f'\tРезультат: {newton_interpolation(x_marked, cord)}')
    print(f'\tОшибка: {abs(newton_interpolation(x_marked, cord) - f(x_marked))}')
    print()
