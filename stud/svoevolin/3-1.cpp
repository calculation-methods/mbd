#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <functional>
#include <algorithm>
#include "matrix.h"

using namespace std;

struct polynomial
{
private:
    vector <double> v;

public:
    polynomial(vector <double> _v = {})
    {
        v = _v;
    }

    double size()
    {
        return v.size();
    }

    double operator[](int idx)
    {
        return v[idx];
    }

    double calculate(double x)
    {
        double res = 0;
        double cur = 1;
        for (int i = 0; i < v.size(); i++)
        {
            res += cur * v[i];
            cur = cur * x;
        }
        return res;
    }
};

ostream& operator<<(ostream& stream, polynomial a)
{
    for (int i = a.size() - 1; i > 0; i--)
        stream << ((i == a.size() - 1) ? a[i] : abs(a[i])) << "*" << "x^" << i << ((a[i - 1] > 0) ? '+' : '-');
    stream << abs(a[0]);
    return stream;
}

vector <double> open_brakets(vector <double> v)
{
    if (v.size() == 1)
        return { v[0], 1 };
    int n = v.size();
    double last = v.back();
    v.erase(--v.end());
    vector <double> res = open_brakets(v);
    vector <double> tmp = res;
    for (int i = 0; i < n; i++)
        res[i] = res[i] * last;
    res.push_back(0);
    for (int i = 1; i <= n; i++)
        res[i] += tmp[i - 1];
    return res;
}

polynomial interpolation_lagrange(vector <double> x, vector <double> y)
{
    int n = x.size();
    vector <double> res(n);
    for (int i = 0; i < n; i++)
    {
        vector <double> v;
        double k = y[i];
        for (int j = 0; j < n; j++)
        {
            if (i == j)
                continue;
            v.push_back(-x[j]);
            k = k / (x[i] - x[j]);
        }
        vector <double> tmp = open_brakets(v);
        for (int j = 0; j < n; j++)
            res[j] += k * tmp[j];
    }
    return polynomial(res);
}

polynomial interpolation_newton(vector <double> x, vector <double> y)
{
    int n = x.size();
    vector <double> res(n);
    res[0] = y[0];
    vector <vector <double>> diff(n - 1, vector <double>(n - 1));
    for (int i = 0; i < n - 1; i++)
        diff[0][i] = (y[i] - y[i + 1]) / (x[i] - x[i + 1]);
    for (int i = 1; i < n - 1; i++)
    {
        for (int j = 0; j < n - 1 - i; j++)
            diff[i][j] = (diff[i - 1][j] - diff[i - 1][j + 1]) / (x[j] - x[j + 1 + i]);
    }
    vector <double> cur;
    for (int i = 0; i < n - 1; i++)
    {
        cur.push_back(-x[i]);
        vector <double> tmp = open_brakets(cur);
        double k = diff[i][0];
        for (int j = 0; j < tmp.size(); j++)
            res[j] += k * tmp[j];
    }
    return polynomial(res);
}


double f1(double x)
{
    return acos(x) + x;
}

int main()
{
    setlocale(LC_ALL, "Rus");
    ofstream fout("answer3-1.txt");
    fout.precision(5);
    fout << fixed;

    vector <double> X = { -0.4, -0.1, 0.2, 0.5 };
    vector <double> Y;
    fout << "Значения функции в точках:\n";
    for (int i = 0; i < X.size(); i++)
    {
        fout << f1(X[i]) << ' ';
        Y.push_back(f1(X[i]));
    }
    polynomial p1 = interpolation_lagrange(X, Y);
    fout << "\nМногочлен Лагранжа: " << p1 << '\n';
    polynomial p2 = interpolation_newton(X, Y);
    fout << "Многочлен Ньютона: " << p2 << '\n';
    fout << "Значения многочленов в точках:\n";
    for (int i = 0; i < X.size(); i++)
        fout << p1.calculate(X[i]) << ' ';

}