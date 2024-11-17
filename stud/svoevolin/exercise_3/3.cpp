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

matrix operator*(double lhs, matrix rhs)
{
    for (int i = 0; i < rhs.rows; i++)
    {
        for (int j = 0; j < rhs.cols; j++)
            rhs[i][j] *= lhs;
    }
    return rhs;
}

matrix operator+(matrix lhs, matrix rhs)
{
    if (lhs.rows != rhs.rows || rhs.cols != lhs.cols)
        return matrix(0, 0);
    matrix res(lhs.rows, lhs.cols);
    for (int i = 0; i < rhs.rows; i++)
    {
        for (int j = 0; j < res.cols; j++)
            res[i][j] = lhs[i][j] + rhs[i][j];
    }
    return res;
}

matrix operator-(matrix lhs, matrix rhs)
{
    if (lhs.rows != rhs.rows || rhs.cols != lhs.cols)
        return matrix(0, 0);
    matrix res(lhs.rows, lhs.cols);
    for (int i = 0; i < rhs.rows; i++)
    {
        for (int j = 0; j < res.cols; j++)
            res[i][j] = lhs[i][j] - rhs[i][j];
    }
    return res;
}

double abs(matrix a)
{
    double mx = 0;
    for (int i = 0; i < a.rows; i++)
    {
        double s = 0;
        for (int j = 0; j < a.cols; j++)
            s += abs(a[i][j]);
        mx = max(mx, s);
    }
    return mx;
}

matrix solve_iteration(matrix a, matrix b, double eps, int& iter)
{
    int n = a.rows;
    matrix alpha(n, n), beta(n, 1);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            alpha[i][j] = -a[i][j] / a[i][i];
        alpha[i][i] = 0;
    }
    for (int i = 0; i < n; i++)
        beta[i][0] = b[i][0] / a[i][i];
    matrix x = beta;
    double m = abs(a);
    double cur = m;
    double epsk = 2 * eps;
    iter = 0;
    while (epsk > eps)
    {
        matrix prev = x;
        x = beta + alpha * x;
        if (m < 1)
            epsk = cur / (1 - m) * abs(x - prev);
        else
            epsk = abs(x - prev);
        cur = cur * m;
        iter++;
    }
    return x;
}

matrix solve_seidel(matrix a, matrix b, double eps, int& iter)
{
    int n = a.rows;
    matrix alpha(n, n), beta(n, 1);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            alpha[i][j] = -a[i][j] / a[i][i];
        alpha[i][i] = 0;
    }
    for (int i = 0; i < n; i++)
        beta[i][0] = b[i][0] / a[i][i];
    matrix x = beta;
    double m = abs(a);
    double cur = m;
    double epsk = 2 * eps;
    iter = 0;
    while (epsk > eps)
    {
        matrix prev = x;
        for (int i = 0; i < n; i++)
        {
            double cur = beta[i][0];
            for (int j = 0; j < n; j++)
                cur += alpha[i][j] * x[j][0];
            x[i][0] = cur;
        }
        if (m < 1)
            epsk = cur / (1 - m) * abs(x - prev);
        else
            epsk = abs(x - prev);
        cur = cur * m;
        iter++;
    }
    return x;
}

int main()
{
    ofstream fout("answer.txt");
    fout.precision(2);
    fout << fixed;
    ifstream fina("matrix3.txt"), finb("column3.txt");
    matrix a(4, 4), b(4, 1);
    fina >> a;
    finb >> b;
    int iter = 0;
    fout << "Решение системы методом итераций с e=0.01:\n" << solve_iteration(a, b, 0.001, iter) << "Количество итераций: ";
    fout << iter;
    fout << "\nРешение системы методом Зейделя с e=0.01:\n" << solve_seidel(a, b, 0.001, iter) << "Количество итераций: ";
    fout << iter;
}