from typing import Callable


def func(x: float) -> float:
    return x ** 3 + x ** 2 - 2 * x - 1


def newton_method(x_0: float, EPS: float, f: Callable) -> tuple[float, int]:
    """
    Функция f(x): x^3 + x^2 - 2x - 1
    Первая производная: 3x^2 + 2x - 2
    Вторая производная: 6x + 2

    Условие сходимости:
    f(x_0)*f''(x_0) > 0

    Тогда для заданной функции подходит начальное значение:
    x_0 = 2 (выбрано случайно)
    """
    def d_func(x: float) -> float:
        return 3 * (x ** 2) + 2 * x - 2

    def eps(val1: float, val2: float) -> float:
        return abs(val1 - val2)

    if func(x_0) == 0:
        return x_0, 0
    x_next, x_curr, k = x_0 - f(x_0)/d_func(x_0), x_0, 1
    while eps(x_curr, x_next) >= EPS:
        k += 1
        x_next, x_curr = x_next - f(x_next)/d_func(x_next), x_next
    return x_next, k


def simple_iterations_method(x_0: float, a: float, b: float, q: float, EPS: float, f: Callable) -> tuple[float, int]:
    """
    Функция f(x): x^3 + x^2 - 2x - 1
    Уравнение с выделенным членом (x = phi(x)): x = (x^3 + x^2 - 1)/2
    phi(x) = (x^3 + x^2 - 1)/2
    phi'(x) = 1.5*x^2 + x

    Условие сходимости:
    1) phi(x)∈[a, b] ∀x∈[a, b]
    2) ∃ q: |phi'(x)|<=q<1 ∀x∈(a, b)

    Тогда для заданной функции подходят параметры:
    q = 0.8
    [a, b] = [-1/3, (-1 + (1 + 1.5*4*q)**0.5)/3]
    x_0 = -0.25
    """
    def phi(x: float) -> float:
        return (x ** 3 + x ** 2 - 1)/2

    def eps(val1: float, val2: float) -> float:
        return q*abs(val1 - val2)/(1-q)

    if func(x_0) == 0:
        return x_0, 0

    x_next, x_curr, k = x_0 + 2 * EPS, x_0, 0
    while eps(x_curr, x_next) >= EPS:
        k += 1
        x_next, x_curr = phi(x_next), x_next
    return x_next, k


if __name__ == "__main__":
    deviation = 1e-9
    q = 0.8

    result, number_of_iterations = newton_method(x_0=2, EPS=deviation, f=func)
    print('\nМетод Ньютона')
    print(f'Результат (x_k): {result}\nКоличество итераций (k): {number_of_iterations}')
    print(f'Значение функции при данном корне x_k: {func(result)}')
    print()

    parameters = {
        "x_0": -0.25,
        "a": -1 / 3,
        "b": (-1 + (1 + 1.5*4*q)**0.5)/3,
        "q": q,
        "EPS": deviation,
        "f": func
    }
    result, number_of_iterations = simple_iterations_method(**parameters)
    print('\nМетод простых итераций')
    print(f'Результат (x_k): {result}\nКоличество итераций (k): {number_of_iterations}')
    print(f'Значение функции при данном корне x_k: {func(result)}')
    print()
