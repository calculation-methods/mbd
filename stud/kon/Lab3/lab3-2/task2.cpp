#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;

double spline(vector<double> x, vector<double> y, double X, int n, vector<double>& a, vector<double>& b, vector<double>& c, vector<double>& d) {

    vector<double> h(n - 1);
    vector<double> p(n - 1);
    vector<double> l(n);
    vector<double> mu(n - 1);
    vector<double> z(n);

    for (int i = 0; i < n - 1; ++i) {
        h[i] = x[i + 1] - x[i];
    }

    for (int i = 1; i < n - 1; ++i) {
        p[i] = 3 * (y[i + 1] - y[i]) / h[i] - 3 * (y[i] - y[i - 1]) / h[i - 1];
    }

    l[0] = 1;
    mu[0] = 0;
    z[0] = 0;

    for (int i = 1; i < n - 1; ++i) {
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (p[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    l[n - 1] = 1;
    z[n - 1] = 0;

    for (int i = 0; i < n - 1; ++i){
        a[i] = y[i];
    }

    for (int j = n - 2; j >= 0; --j) {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
        d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
    }

    int index = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (x[i] <= X && X <= x[i + 1]) {
            index = i;
            break;
        }
    }
    
    double res = a[index] + b[index] * (X - x[index]) + c[index] * pow((X- x[index]), 2) + d[index] * pow((X - x[index]), 3);

    return res;
}

int main() {
    int n = 5;
    vector<double> x = {0.0, 0.5, 1.0, 1.5, 2.0};
    double X = 0.8;
    vector<double> y = {0.0, 0.97943, 1.8415, 2.4975, 2.9093};
    vector<double> a(n-1, 0), b(n-1, 0), c(n-1, 0), d(n-1, 0);

    ofstream fout("output.txt");

    double res = spline(x, y, X, n, a, b, c, d);

    fout << "Коэффициенты:" << endl;
    for (int i = 0; i < n-1; ++i) {
        fout << a[i] << " " << b[i] << " " << c[i] << " " << d[i] << endl;
    }

    fout << "Значение функции в точке X: " << res << endl;

    return 0;
}