#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <functional>
#include <algorithm>
#include "matrix.h"

using namespace std;
using equation = function <double(vector <double>)>;
using func = function <double(double)>;

double runge_romberg(double i1, double i2, double h1, double h2, double p)
{
    double k = h2 / h1;
    return (i1 - i2) / (pow(k, p) - 1);
}

vector <equation> make_system(equation dy, int k)
{
    vector <equation> res(k);
    for (int i = 0; i < k - 1; i++)
    {
        auto f = [=](vector <double> x) mutable
        {
            return x[i + 2];
        };
        res[i] = f;
    }
    res[k - 1] = dy;
    return res;
}

vector <vector <double>> explicit_euler(vector <equation> dy, vector <double> yk, double a, double b, double h)
{
    int n = dy.size();
    vector <vector <double>> res(n);
    for (int i = 0; i < n; i++)
        res[i].push_back(yk[i]);
    for (double x = a; x <= b - h; x += h)
    {
        vector <double> args;
        args.push_back(x);
        for (int i = 0; i < n; i++)
            args.push_back(res[i].back());
        for (int i = 0; i < n; i++)
            res[i].push_back(res[i].back() + h * dy[i](args));
    }
    return res;
}

vector <vector <double>> improved_euler(vector <equation> dy, vector <double> yk, double a, double b, double h)
{
    int n = dy.size();
    vector <vector <double>> tmp(n);
    vector <vector <double>> res(n);
    for (int i = 0; i < n; i++)
    {
        tmp[i].push_back(yk[i]);
        res[i].push_back(yk[i]);
    }
    int k = 0;
    for (double x = a; x <= b - h / 2; x += h / 2)
    {
        vector <double> args;
        args.push_back(x);
        for (int i = 0; i < n; i++)
            args.push_back(tmp[i].back());
        for (int i = 0; i < n; i++)
        {
            tmp[i].push_back(tmp[i][tmp[i].size() - 1 - k % 2] + 0.5 * (k % 2 + 1) * h * dy[i](args));
            if (k % 2)
                res[i].push_back(tmp[i].back());
        }
        k++;
    }
    return res;
}

vector <double> make_args(double x, const vector <double>& y, double add_x, double add_y)
{
    vector <double> res;
    res.push_back(x + add_x);
    for (int i = 0; i < y.size(); i++)
        res.push_back(y[i] + add_y);
    return res;
}

vector <vector <double>> runge_kutta(vector <equation> dy, vector <double> yk, double l, double r, double h)
{
    // для 4 порядка точности
    int p = 4;
    vector <double> a = { 0, 0, 0.5, 0.5, 1 };
    vector <vector <double>> b = { {}, {0}, {0, 0.5}, {0, 0, 0.5}, {0, 0, 0, 0.5} };
    vector <double> c = { 0, 1. / 6, 1. / 3, 1. / 3, 1. / 6 };
    //
    int n = dy.size();
    vector <vector <double>> res(n);
    for (int i = 0; i < n; i++)
        res[i].push_back(yk[i]);
    vector <double> K(p + 1);
    for (double x = l; x <= r - h; x += h)
    {
        vector <double> y;
        for (int idx = 0; idx < n; idx++)
            y.push_back(res[idx].back());
        for (int idx = 0; idx < n; idx++)
        {
            K[1] = h * dy[idx](make_args(x, y, 0, 0));
            for (int i = 2; i <= p; i++)
            {
                double add = 0;
                for (int j = 1; j <= i - 1; j++)
                    add += b[i][j] * K[j];
                K[i] = h * dy[idx](make_args(x, y, a[i] * h, add));
            }
            double delta = 0;
            for (int i = 1; i <= p; i++)
                delta += c[i] * K[i];
            res[idx].push_back(res[idx].back() + delta);
        }
    }
    return res;
}

vector <vector <double>> adams(vector <equation> dy, vector <double> yk, double a, double b, double h)
{
    int n = dy.size();
    vector <vector <double>> y = runge_kutta(dy, yk, a, b, h);
    int m = y[0].size();
    vector <vector <double>> res(n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < 4; j++)
            res[i].push_back(y[i][j]);
    }
    for (int k = 4; k < m; k++)
    {
        vector <vector <double>> args(4);
        for (int j = 0; j < 4; j++)
        {
            args[j].push_back(a + h * (k - j - 1));
            for (int i = 0; i < n; i++)
                args[j].push_back(res[i][k - j - 1]);
        }
        for (int i = 0; i < n; i++)
        {
            double delta = 55 * dy[i](args[0]) - 59 * dy[i](args[1]) + 37 * dy[i](args[2]) - 9 * dy[i](args[3]);
            res[i].push_back(res[i].back() + (h / 24) * delta);
        }
    }
    return res;
}

vector <double> shooting(equation ddy, double a, double b, vector <double> alpha, vector <double> beta, double ya, double yb, double h)
{
    double eps = 0.00000001;
    vector <equation> v = make_system(ddy, 2);
    auto phi = [&](double n)
    {
        vector <double> args;
        if (abs(beta[0]) > eps)
            args = { n, (ya - alpha[0] * n) / beta[0] };
        else
            args = { ya / alpha[0], n };
        vector <vector <double>> yk = runge_kutta(v, args, a, b, h);
        return alpha[1] * yk[0].back() + beta[1] * yk[1].back() - yb;
    };
    double n0 = 10, n1 = -1;
    double phi0 = phi(n0);
    double phi1 = phi(n1);
    double n;
    while (true)
    {
        n = n1 - ((n1 - n0) / (phi1 - phi0)) * phi1;
        double phij = phi(n);
        if (abs(phij) < eps)
            break;
        n0 = n1;
        n1 = n;
        phi0 = phi1;
        phi1 = phij;
    }
    vector <double> args;
    if (abs(beta[0]) > eps)
        args = { n, (ya - alpha[0] * n) / beta[0] };
    else
        args = { ya / alpha[0], n };
    vector <vector <double>> res = runge_kutta(v, args, a, b, h);
    return res[0];
}

vector <double> finite_difference(func f, func p, func q, double a, double b, vector <double> alpha, vector <double> beta, double ya, double yb, double h)
{
    vector <double> x;
    for (int i = 0; a + i * h < b; i++)
        x.push_back(a + i * h);
    int n = x.size();
    matrix A(n + 1, n + 1);
    matrix B(n + 1, 1);
    A[0][0] = alpha[0] * h - beta[0];
    A[0][1] = beta[0];
    B[0][0] = h * ya;
    for (int i = 1; i <= n - 1; i++)
    {
        A[i][i + 1] = 1 + p(x[i]) * h / 2;
        A[i][i] = -2 + h * h * q(x[i]);
        A[i][i - 1] = 1 - p(x[i]) * h / 2;
        B[i][0] = h * h * f(x[i]);
    }
    A[n][n - 1] = -beta[1];
    A[n][n] = alpha[1] * h + beta[1];
    B[n][0] = h * yb;
    matrix sol = solve_tridiagonal(A, B);
    vector <double> res;
    for (int i = 0; i < n + 1; i++)
        res.push_back(sol[i][0]);
    return res;
}

int main()
{
    setlocale(LC_ALL, "Rus");
    ofstream fout("answer4-2.txt");
    fout.precision(8);
    fout << fixed;

    auto ddy = [](vector <double> x)
    {
        return (2 * x[1] - 2 * (x[0] + 1) * x[2]) / (x[0] * (2 * x[0] + 1));
    };
    double h1 = 0.1;
    vector <double> y1 = shooting(ddy, 1, 3, { 0, 1 }, { 1, -1 }, 0, 31. / 9, h1);

    auto f = [](double x) { return 0; };
    auto p = [](double x) { return 2 * (x + 1) / (x * (2 * x + 1)); };
    auto q = [](double x) { return -2 / (x * (2 * x + 1)); };
    vector <double> y2 = finite_difference(f, p, q, 1, 3, { 0, 1 }, { 1, -1 }, 0, 31. / 9, h1);
    fout << "Стрельба" << "\t" << "Разности" << '\t' << "Точное" << endl;
    for (int i = 0; i < y1.size(); i++)
    {
        double x = 1 + i * h1;
        fout << y1[i] << '\t' << y2[i] << '\t' << (x + 1 + 1 / x) << endl;
    }
    double h2 = 0.05;
    vector <double> y12 = shooting(ddy, 1, 3, { 0, 1 }, { 1, -1 }, 0, 31. / 9, h2);
    vector <double> y22 = finite_difference(f, p, q, 1, 3, { 0, 1 }, { 1, -1 }, 0, 31. / 9, h2);
    fout << "\nПогрешности методом Рунге-Ромберга\n";
    fout << "Стрельба" << "\t" << "Разности" << endl;
    for (int i = 0; i < y1.size(); i++)
    {
        fout << runge_romberg(y1[i], y12[2 * i], h1, h2, 4) << '\t'
            << runge_romberg(y2[i], y22[2 * i], h1, h2, 2) << endl;
    }
}