#pragma once
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

istream& operator>>(istream& stream, matrix& a)
{
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
            stream >> a[i][j];
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

vector <int> swp;

pair <matrix, matrix> lu_decomposition(matrix a)
{
    int n = a.rows;
    matrix l(n, n);
    swp = vector <int>(0);
    for (int k = 0; k < n; k++)
    {
        matrix prev = a;
        int idx = k;
        for (int i = k + 1; i < n; i++)
        {
            if (abs(prev[idx][k]) < abs(prev[i][k]))
                idx = i;
        }
        swap(prev[k], prev[idx]);
        swap(a[k], a[idx]);
        swap(l[k], l[idx]);
        swp.push_back(idx);
        for (int i = k + 1; i < n; i++)
        {
            double h = prev[i][k] / prev[k][k];
            l[i][k] = h;
            for (int j = k; j < n; j++)
                a[i][j] = prev[i][j] - h * prev[k][j];

        }
    }
    for (int i = 0; i < n; i++)
        l[i][i] = 1;
    return { l, a };
}

matrix solve_triag(matrix a, matrix b, bool up)
{
    int n = a.rows;
    matrix res(n, 1);
    int d = up ? -1 : 1;
    int first = up ? n - 1 : 0;
    for (int i = first; i < n && i >= 0; i += d)
    {
        res[i][0] = b[i][0];
        for (int j = 0; j < n; j++)
        {
            if (i != j)
                res[i][0] -= a[i][j] * res[j][0];
        }
        res[i][0] = res[i][0] / a[i][i];
    }
    return res;
}

matrix solve_gauss(pair <matrix, matrix> lu, matrix b)
{
    for (int i = 0; i < swp.size(); i++)
        swap(b[i], b[swp[i]]);
    matrix z = solve_triag(lu.first, b, false);
    matrix x = solve_triag(lu.second, z, true);
    //for (int i = 0; i < swp.size(); i++)
        //swap(x[i], x[swp[i]]);
    return x;
}

matrix inverse(matrix a)
{
    int n = a.rows;
    matrix b(n, 1);
    pair <matrix, matrix> lu = lu_decomposition(a);
    matrix res(n, n);
    for (int i = 0; i < n; i++)
    {
        b[max(i - 1, 0)][0] = 0;
        b[i][0] = 1;
        matrix col = solve_gauss(lu, b);
        for (int j = 0; j < n; j++)
            res[j][i] = col[j][0];
    }
    return res;
}

double determinant(matrix a)
{
    int n = a.rows;
    pair <matrix, matrix> lu = lu_decomposition(a);
    double det = 1;
    for (int i = 0; i < n; i++)
        det *= lu.second[i][i];
    return det;
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

matrix solve_iteration(matrix a, matrix b, double eps)
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
    double epsk = 2 * eps;
    while (epsk > eps)
    {
        matrix prev = x;
        x = beta + alpha * x;
        if (m < 1)
            epsk = m / (1 - m) * abs(x - prev);
        else
            epsk = abs(x - prev);
    }
    return x;
}

matrix solve_seidel(matrix a, matrix b, double eps)
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
    double m = abs(alpha);
    double epsk = 2 * eps;
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
            epsk = m / (1 - m) * abs(x - prev);
        else
            epsk = abs(x - prev);
    }
    return x;
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

double sign(double x)
{
    return x > 0 ? 1 : -1;
}

pair <matrix, matrix> qr_decomposition(matrix a)
{
    int n = a.rows;
    matrix e(n, n);
    for (int i = 0; i < n; i++)
        e[i][i] = 1;
    matrix q = e;
    for (int i = 0; i < n - 1; i++)
    {
        matrix v(n, 1);
        double s = 0;
        for (int j = i; j < n; j++)
            s += a[j][i] * a[j][i];
        v[i][0] = a[i][i] + sign(a[i][i]) * sqrt(s);
        for (int j = i + 1; j < n; j++)
            v[j][0] = a[j][i];
        matrix h = e - (2.0 / double(transposition(v) * v)) * (v * transposition(v));
        q = q * h;
        a = h * a;
    }
    return { q, a };
}

vector <cmd> qr_eigenvalues(matrix a, double eps)
{
    int n = a.rows;
    vector <cmd> prev(n);
    while (true)
    {
        pair <matrix, matrix> p = qr_decomposition(a);
        a = p.second * p.first;
        vector <cmd> cur;
        for (int i = 0; i < n; i++)
        {
            if (i < n - 1 && abs(a[i + 1][i]) > 1e-7)
            {
                double b = -(a[i][i] + a[i + 1][i + 1]);
                double c = a[i][i] * a[i + 1][i + 1] - a[i][i + 1] * a[i + 1][i];
                double d = b * b - 4 * c;
                cmd sgn = (d > 0) ? cmd(1, 0) : cmd(0, 1);
                d = sqrt(abs(d));
                cur.push_back(0.5 * (-b - sgn * d));
                cur.push_back(0.5 * (-b + sgn * d));
                i++;
            }
            else
                cur.push_back(a[i][i]);
        }
        bool ok = true;
        for (int i = 0; i < n; i++)
            ok = ok && abs(cur[i] - prev[i]) < eps;
        if (ok)
            break;
        prev = cur;
    }
    return prev;
}
