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


if __name__ == "__main__":
    x = [(3*i - 7)/10 for i in range(6)]
    y = [2.3462, 1.9823, 1.671, 1.3694, 1.0472, 0.6435]

    n = len(x)

    matr = Matrix([
        [n, sum(x)],
        [sum(x), sum([val**2 for val in x])]
    ])

    root = Matrix([
        [sum(y)],
        [sum([y[i]*x[i] for i in range(n)])]
    ])

    f1_coeffs = tuple(map(lambda vect: vect[0], calculate_decisions(matr, root)))
    print(f'F1(x) = {f1_coeffs[0]}{" + " if f1_coeffs[1] >=0 else " - "}{abs(f1_coeffs[1])}*x')
    print(f'Loss = {sum([(f1_coeffs[0] + f1_coeffs[1]*x[i] - y[i])**2 for i in range(n)])}')
    print()

    matr = Matrix([
        [n, sum(x), sum([val ** 2 for val in x])],
        [sum(x), sum([val ** 2 for val in x]), sum([val ** 3 for val in x])],
        [sum([val ** 2 for val in x]), sum([val ** 3 for val in x]), sum([val ** 4 for val in x])]
    ])

    root = Matrix([
        [sum(y)],
        [sum([y[i] * x[i] for i in range(n)])],
        [sum([y[i] * x[i]**2 for i in range(n)])]
    ])

    f2_coeffs = tuple(map(lambda vect: vect[0], calculate_decisions(matr, root)))
    print(f'F2(x) = {f2_coeffs[0]}{" + " if f2_coeffs[1] >= 0 else " - "}{abs(f2_coeffs[1])}*x{" + " if f2_coeffs[2] >= 0 else " - "}{abs(f2_coeffs[2])}*x^2')
    print(f'Loss = {sum([(f2_coeffs[0] + f2_coeffs[1] * x[i] + f2_coeffs[2] * x[i]**2 - y[i]) ** 2 for i in range(n)])}')
    print()
