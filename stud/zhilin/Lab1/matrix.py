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
    res: list[list[None | int | float]] = [[None]*m2 for _ in range(n1)]
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
    res = [[matrix1[i][j]+matrix2[i][j] for j in range(matrix1.m)] for i in range(matrix1.n)]
    return Matrix(res)

