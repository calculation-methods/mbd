#include <vector>
#include <ccomplex>
#include <fstream>

using namespace std;

struct matrix
{
    int n = 0, m = 0;
    vector <vector <double>> data;

    matrix() {}
    matrix(int _n, int _m)
    {
        n = _n;
        m = _m;
        data = vector<vector<double>>(n, vector <double>(m));
    }

    vector <double>& operator[](int row)
    {
        return data[row];
    }

    operator double()
    {
        return data[0][0];
    }
};

// Перемножение матриц
matrix operator*(matrix lhs, matrix rhs)
{
    matrix res(lhs.n, rhs.m);
    for (int i = 0; i < res.n; i++)
        for (int j = 0; j < res.m; j++)
        {
            res[i][j] = 0;
            for (int k = 0; k < lhs.m; k++)
                res[i][j] += lhs[i][k] * rhs[k][j];
        }
    return res;
}

// Умножение матрицы на число
matrix operator*(double lhs, matrix rhs)
{
    for (int i = 0; i < rhs.n; i++)
        for (int j = 0; j < rhs.m; j++)
            rhs[i][j] *= lhs;
    return rhs;
}

// Сложение матриц
matrix operator+(matrix lhs, matrix rhs)
{
    matrix res(lhs.n, lhs.m);
    for (int i = 0; i < rhs.n; i++)
        for (int j = 0; j < res.m; j++)
            res[i][j] = lhs[i][j] + rhs[i][j];
    return res;
}

// Вычетание матриц
matrix operator-(matrix lhs, matrix rhs)
{
    matrix res(lhs.n, lhs.m);
    for (int i = 0; i < rhs.n; i++)
        for (int j = 0; j < res.m; j++)
            res[i][j] = lhs[i][j] - rhs[i][j];
    return res;
}

// Вывод матрицы
ostream& operator<<(ostream& stream, matrix _matrix)
{
    for (int i = 0; i < _matrix.n; i++)
    {
        for (int j = 0; j < _matrix.m; j++)
            stream << _matrix[i][j] << " ";
        stream << endl;
    }
    return stream;
}

// Чтение матрицы
istream& operator>>(istream& stream, matrix& _matrix)
{
    for (int i = 0; i < _matrix.n; i++)
        for (int j = 0; j < _matrix.m; j++)
            stream >> _matrix[i][j];
    return stream;
}

// Транспонирование матрицы
matrix transposition(matrix _matrix)
{
    matrix res(_matrix.m, _matrix.n);
    for (int i = 0; i < _matrix.n; i++)
        for (int j = 0; j < _matrix.m; j++)
            res[j][i] = _matrix[i][j];
    return res;
}