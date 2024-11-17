class Vector:
    def __init__(self, values: list[None | int | float]):
        self.values = [value for value in values]
        self.n = len(values)

    def __str__(self) -> str:
        return self.values.__str__()

    def copy(self):
        return Vector([value for value in self.values])

    def __len__(self) -> int:
        return self.values.__len__()

    def __getitem__(self, index: int):
        return self.values[index]

    def __setitem__(self, index: int, value: int | float):
        self.values[index] = value

    def upload_values(self, values: list[list[None | int | float]]):
        self.values = [value for value in values]


class Matrix:
    def __init__(self, values: list[list[None | int | float] | Vector]):
        if type(values[0]) == Vector:
            self.values = [vector.copy() for vector in values]
        else:
            self.values = [Vector(row) for row in values]
        self.n: int = len(self.values)
        self.m: int = len(self.values[0])

    def __str__(self) -> str:
        m = '\n'.join([row.__str__() for row in self.values])
        return f"Shape = {self.get_shape()} \n{m}"

    def get_shape(self) -> tuple[int, int]:
        return self.n, self.m

    def copy(self):
        return Matrix([vect.copy() for vect in self.values])

    def upload_values(self, values: list[list[None | int | float]]):
        self.values = [[val for val in row] for row in values]

    def transposed(self):
        res = [[None] * self.n for _ in range(self.m)]
        for i in range(self.n):
            for j in range(self.m):
                res[j][i] = self.values[i][j]
        return Matrix(res)

    def __setitem__(self, index: int, value: Vector):
        self.values[index] = value

    def __getitem__(self, index: int):
        return self.values[index]


def multiple_matrix(matrix1: Matrix, matrix2: Matrix) -> None | Matrix:
    n1, m1 = matrix1.get_shape()
    n2, m2 = matrix2.get_shape()
    if m1 != n2:  # TODO: implement custom Error for matrices
        print("Incorrect shapes of matrices")
        return
    res: list[list[None | int | float]] = [[None] * m2 for _ in range(n1)]
    n, m, h = n1, m2, m1
    del n1, n2, m1, m2
    for i in range(n):
        for j in range(m):
            cntr = 0
            for k in range(h):
                cntr += matrix1.values[i][k] * matrix2.values[k][j]
            res[i][j] = cntr

    return Matrix(res)


def plus_matrix(matrix1: Matrix, matrix2: Matrix) -> Matrix:
    res = [[matrix1[i][j] + matrix2[i][j] for j in range(matrix1.m)] for i in range(matrix1.n)]
    return Matrix(res)


def tridiagonal_algorithm(coefficients: Matrix, results: Matrix) -> Matrix:
    if coefficients.n < 3 or coefficients.m < 3:
        print("Incorrect shapes of matrices")
        raise Exception
    a, b, c, d = 0, coefficients[0][0], coefficients[0][1], results.values[0][0]
    P, Q = [0] * coefficients.m, [0] * coefficients.m

    P[0], Q[0] = -c / b, d / b
    for i in range(1, coefficients.n - 1):
        a, b, c, d = coefficients[i][i - 1], coefficients[i][i], coefficients[i][i + 1], results[i][0]
        P[i] = -c / (b + a * P[i - 1])
        Q[i] = (d - a * Q[i - 1]) / (b + a * P[i - 1])
    a, b, c, d = coefficients[-1][-2], coefficients[-1][-1], 0, results[-1][0]
    Q[-1] = (d - a * Q[-2]) / (b + a * P[-2])

    decisions = [0] * results.n
    decisions[-1] = Q[-1]
    for i in range(decisions.__len__() - 2, -1, -1):
        decisions[i] = P[i] * decisions[i + 1] + Q[i]

    return Matrix([[i] for i in decisions])


if __name__ == "__main__":
    x_marked = 0.1
    x = [-0.4, -0.1, 0.2, 0.5, 0.8]
    y = [1.9823, 1.6710, 1.3694, 1.0472, 0.64350]

    h = [0.0] + [x[i + 1] - x[i] for i in range(4)]
    n = len(x) - 1

    matr = Matrix(
        [[2 * (h[1] + h[2]), h[2], 0]] +
        [[h[i - 1], 2 * (h[i - 1] + h[i]), h[i]] for i in range(3, n)] +
        [[0, h[n - 1], 2 * (h[n - 1] + h[n])]]
    )

    root = Matrix([[3 * ((y[i + 2] - y[i + 1]) / h[i + 2] - (y[i + 1] - y[i]) / h[i + 1])] for i in range(n - 1)])

    coeff_a = y[:-1]
    coeff_c = [0] + [val[0] for val in tridiagonal_algorithm(matr, root)]
    coeff_b = [(y[i] - y[i - 1]) / h[i] - h[i] * (coeff_c[i] + 2 * coeff_c[i - 1]) / 3 for i in range(1, n)] + [
        (y[n] - y[n - 1]) / h[n] - 2 * h[n] * coeff_c[n - 1] / 3]
    coeff_d = [(coeff_c[i + 1] - coeff_c[i]) / (3 * h[i + 1]) for i in range(n - 1)] + [-coeff_c[n - 1] / (3 * h[n])]

    for i in range(n):
        if x[i] <= x_marked <= x[i + 1]:
            res = coeff_a[i] + coeff_b[i]*(x_marked-x[i]) + coeff_c[i]*(x_marked-x[i])**2 + coeff_d[i]*(x_marked-x[i])**3
            print(f'Result = {res}')
            break
    else:
        print("Incorrect value")
