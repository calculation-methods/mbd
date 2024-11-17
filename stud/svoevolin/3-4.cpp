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

struct cubic_spline
{
private:
    vector <polynomial> p;
    vector <double> xn;

public:
    cubic_spline(vector <double> _xn = { 0 }, vector <polynomial> _p = {})
    {
        if (_p.size() == _xn.size() - 1)
        {
            p = _p;
            xn = _xn;
        }
        else
        {
            p = {};
            xn = { 0 };
        }
    }

    int size()
    {
        return p.size();
    }

    pair <pair <double, double>, polynomial> operator[](int idx)
    {
        return { {xn[idx], xn[idx + 1]}, p[idx] };
    }

    double calculate(double x)
    {
        int idx = upper_bound(xn.begin(), xn.end(), x) - xn.begin();
        idx = min(idx, (int)xn.size() - 1) - 1;
        idx = max(0, idx);
        return p[idx].calculate(x);
    }
};

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

polynomial derivative(polynomial p)
{
    int n = p.size();
    vector <double> res;
    for (int i = 1; i < n; i++)
        res.push_back(i * p[i]);
    return polynomial(res);
}

function <double(double)> derivative(vector <double> x, vector <double> y, int m)
{
    int n = x.size();
    vector <pair <double, double>> xy(n);
    for (int i = 0; i < n; i++)
        xy[i] = { x[i], y[i] };
    sort(xy.begin(), xy.end());
    for (int i = 0; i < n; i++)
    {
        x[i] = xy[i].first;
        y[i] = xy[i].second;
    }
    vector <polynomial> p(n - m);
    for (int i = 0; i < n - m; i++)
    {
        vector <double> xm, ym;
        for (int j = 0; j < m + 1; j++)
        {
            xm.push_back(x[i + j]);
            ym.push_back(y[i + j]);
        }
        p[i] = interpolation_lagrange(xm, ym);
        for (int j = 0; j < m; j++)
            p[i] = derivative(p[i]);
    }
    auto res = [=](double _x) mutable
    {
        int idx = lower_bound(x.begin() + 1, x.end() - m, _x) - x.begin();
        //int idx = upper_bound(x.begin() + 1, x.end() - m, _x) - x.begin();
        idx = idx - 1;
        return p[idx].calculate(_x);
    };
    return res;
}

int main()
{
    setlocale(LC_ALL, "Rus");
    ofstream fout("answer3-4.txt");
    fout.precision(5);
    fout << fixed;
    function <double(double)> df = derivative({-1, 0, 1, 2, 3}, {1.3562, 1.5708, 1.7854, 2.4636, 3.3218}, 1);
    function <double(double)> ddf = derivative({-1, 0, 1, 2, 3}, {1.3562, 1.5708, 1.7854, 2.4636, 3.3218}, 2);
    fout << "Значение первой производной: " << df(1);
    fout << "\nЗначение второй производной: " << ddf(1);
}