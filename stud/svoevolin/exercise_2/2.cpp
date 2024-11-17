#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

struct matrix
{
    int rows = 0, cols = 0;
    vector <vector <double>> v;

    matrix() {}
    matrix(int _rows, int _cols)
    {
        rows = _rows;
        cols = _cols;
        v = vector <vector <double>>(rows, vector <double>(cols));
    }

    vector <double>& operator[](int row)
    {
        return v[row];
    }

    operator double()
    {
        return v[0][0];
    }
};

istream& operator>>(istream& stream, matrix& a)
{
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
            stream >> a[i][j];
    }
    return stream;
}

ostream& operator<<(ostream& stream, matrix a)
{
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
            stream << a[i][j] << ' ';
        stream << '\n';
    }
    return stream;
}


matrix solve_tridiagonal(matrix& a, matrix& b)
{
    int n = a.rows;
    vector <double> p(n), q(n);
    p[0] = -a[0][1] / a[0][0];
    q[0] = b[0][0] / a[0][0];
    for (int i = 1; i < n; i++)
    {
        if (i != n - 1)
            p[i] = -a[i][i + 1] / (a[i][i] + a[i][i - 1] * p[i - 1]);
        else
            p[i] = 0;
            q[i] = (b[i][0] - a[i][i - 1] * q[i - 1]) / (a[i][i] + a[i][i - 1] * p[i - 1]);
    }
    matrix res(n, 1);
    res[n - 1][0] = q[n - 1];
    for (int i = n - 2; i >= 0; i--)
        res[i][0] = p[i] * res[i + 1][0] + q[i];
    return res;
}

int main() 
{
    ofstream fout("answer.txt");
    fout.precision(2);
    fout << fixed;
    ifstream fina("matrix2.txt"), finb("column2.txt");
    matrix a(5, 5), b(5, 1);
    fina >> a;
    finb >> b;
    fout << "Решение системы:\n" << solve_tridiagonal(a, b);
}
