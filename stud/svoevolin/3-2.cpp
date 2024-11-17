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

ostream& operator<<(ostream& stream, cubic_spline a)
{
    for (int i = 0; i < a.size(); i++)
        stream << "[" << a[i].first.first << ";" << a[i].first.second << "] " << a[i].second << endl;
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

cubic_spline make_spline(vector <double> x, vector <double> y)
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
    n--;
    vector <double> a(n), b(n), c(n), d(n);
    vector <double> h(n);
    for (int i = 0; i < n; i++)
        h[i] = x[i + 1] - x[i];
    matrix A(n - 1, n - 1), B(n - 1, 1);
    for (int i = 0; i < n - 1; i++)
    {
        if (i > 0)
            A[i][i - 1] = h[i];
        A[i][i] = 2 * (h[i] + h[i + 1]);
        if (i < n - 2)
            A[i][i + 1] = h[i + 1];
        B[i][0] = 3 * ((y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i]);
    }
    matrix s = solve_tridiagonal(A, B);
    for (int i = 1; i < n; i++)
        c[i] = s[i - 1][0];
    for (int i = 0; i < n; i++)
        a[i] = y[i];
    for (int i = 0; i < n - 1; i++)
        b[i] = (y[i + 1] - y[i]) / h[i] - 1. / 3. * (c[i + 1] + 2 * c[i]);
    b[n - 1] = (y[n] - y[n - 1]) / h[n - 1] - 2. / 3. * h[n - 1] * c[n - 1];
    for (int i = 0; i < n - 1; i++)
        d[i] = (c[i + 1] - c[i]) / (3 * h[i]);
    d[n - 1] = -c[n - 1] / (3 * h[n - 1]);
    vector <polynomial> vp(n);
    for (int i = 0; i < n; i++)
    {
        vector <double> res(4);
        res[0] = a[i];
        vector <double> tmp;
        for (int j = 1; j < 4; j++)
        {
            tmp.push_back(-x[i]);
            vector <double> v = open_brakets(tmp);
            for (int k = 0; k < v.size(); k++)
                res[k] += v[k];
        }
        vp[i] = polynomial(res);
    }
    return cubic_spline(x, vp);
}

int main()
{
    setlocale(LC_ALL, "Rus");
    ofstream fout("answer3-2.txt");
    fout.precision(5);
    fout << fixed;

    
    vector <double> X = { -0.4, -0.1, 0.2, 0.5, 0.8 };
    vector <double> Y = { 1.5823, 1.5710, 1.5694, 1.5472, 1.4435 };
    cubic_spline cs = make_spline(X, Y);
    fout << cs;
    fout << "Значение сплайна в точке: " << cs.calculate(0.1);

}