#include <iostream>
#include <vector>
#include <ccomplex>
#include <fstream>

using namespace std;

using cmd = complex <double>;
const double pi = acos(-1);

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

matrix transposition(matrix a)
{
    matrix res(a.cols, a.rows);
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
            res[j][i] = a[i][j];
    }
    return res;
}

matrix operator*(matrix lhs, matrix rhs)
{
    if (lhs.cols != rhs.rows)
        return matrix(0, 0);
    matrix res(lhs.rows, rhs.cols);
    for (int i = 0; i < res.rows; i++) 
    {
        for (int j = 0; j < res.cols; j++)
        {
            res[i][j] = 0;
            for (int k = 0; k < lhs.cols; k++)
                res[i][j] += lhs[i][k] * rhs[k][j];
        }
    }
    return res;
}

pair <matrix, matrix> method_jacobi(matrix a, double eps)
{
    int n = a.rows;
    double epsk = 2 * eps;
    matrix vec(n, n);
    for (int i = 0; i < n; i++)
        vec[i][i] = 1;
    while (epsk > eps)
    {
        int cur_i = 1, cur_j = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < i; j++)
            {
                if (abs(a[cur_i][cur_j]) < abs(a[i][j]))
                {
                    cur_i = i;
                    cur_j = j;
                }
            }
        }
        matrix u(n, n);
        double phi = pi / 4;
        if (abs(a[cur_i][cur_i] - a[cur_j][cur_j]) > 1e-7)
            phi = 0.5 * atan((2 * a[cur_i][cur_j]) / (a[cur_i][cur_i] - a[cur_j][cur_j]));
        for (int i = 0; i < n; i++)
            u[i][i] = 1;
        u[cur_i][cur_j] = -sin(phi);
        u[cur_i][cur_i] = cos(phi);
        u[cur_j][cur_i] = sin(phi);
        u[cur_j][cur_j] = cos(phi);
        vec = vec * u;
        a = transposition(u) * a * u;
        epsk = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < i; j++)
                epsk += a[i][j] * a[i][j];
        }
        epsk = sqrt(epsk);
    }
    matrix val(n, 1);
    for (int i = 0; i < n; i++)
        val[i][0] = a[i][i];
    return { val, vec };
}

int main() {
    ofstream fout("answer.txt");
    fout.precision(2);
    fout << fixed;
    ifstream fin("matrix4.txt");
    matrix a(3, 3);
    fin >> a;
    pair <matrix, matrix> p = method_jacobi(a, 0.1);
    fout << "Собственные значения с e=0.1:\n" << p.first << "\nСобственные векторы с e=0.1:\n" << p.second;
}