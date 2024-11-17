#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;

double f(double x) {
    return (sin(x)+x);
}

pair<vector<double>, double> lagrange(vector<double> x, vector<double> y, double X, int n) {
    double res = 0.0;
    vector<double> coeff(n, 0);
    for (int i = 0; i < n; ++i) {
        coeff[i] = y[i];
        double l = y[i];
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                coeff[i] = coeff[i]/(x[i] - x[j]);
                l *= (X - x[j]) / (x[i] - x[j]);
            }
        }
        res += l;
    }
    return make_pair(coeff, res);
}

pair<vector<double>, double> newton(vector<double> x, vector<double> y, double X, int n) {
    double res = 0.0;
    vector<double> coeff(n, 0);
    vector<vector<double>> f(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        f[i][0] = y[i];
    }
    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < n - i; ++j) {
            f[j][i] = (f[j + 1][i - 1] - f[j][i - 1]) / (x[i + j] - x[j]);
        }
    }
    for (int i = 0; i < n; ++i) {
        coeff[i] = f[0][i];
        double tmp = f[0][i];
        for (int j = 0; j < i; ++j) {
            tmp *= (X - x[j]);
        }
        res += tmp;
    }
    return make_pair(coeff, res);
}

int main(){
    int n=4;
    vector<double> x_a = {0.0, M_PI/6, 2* M_PI/6, 3* M_PI/6};
    vector<double> y_a;
    vector<double> coeff_a(n, 0);
    double X = 1.0, res, delta;
    for (double xi : x_a) {
        y_a.push_back(f(xi));
    }
    coeff_a = lagrange(x_a, y_a, X, n).first;
    res = lagrange(x_a, y_a, X, n).second;

    ofstream fout("output.txt");

    fout << "Коэффициенты:" << endl;
    for (int i = 0; i < n; ++i) {
        fout << coeff_a[i] << endl;
    }

    fout << "Значение многочлена Лагранжа: " << res << endl;
    delta = abs(f(X)-res);
    fout << "Погрешность интерполяции в точке X: " << delta << endl;

    vector<double> x_b = {0.0, M_PI/6, M_PI/4, M_PI/2};
    vector<double> y_b;
    vector<double> coeff_b(n, 0);
    res = 0.0;
    delta = 0.0;
    for (double xi : x_b) {
        y_b.push_back(f(xi));
    }
    coeff_b = newton(x_b, y_b, X, n).first;
    res = newton(x_b, y_b, X, n).second;
    fout << "Коэффициенты:" << endl;
    for (int i = 0; i < n; ++i) {
        fout << coeff_b[i] << endl;
    }

    fout << "Значение многочлена Ньютона: " << res << endl;
    delta = abs(f(X)-res);
    fout << "Погрешность интерполяции в точке X: " << delta << endl;
    return 0;
}