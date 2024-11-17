#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

// Структура для хранения коэффициентов сплайна
struct Spline {
    double a, b, c, d, x;
};

// Функция для вычисления сплайна
void cubicSpline(const vector<double>& x, const vector<double>& y, vector<Spline>& splines) {
    int n = x.size();
    vector<double> h(n - 1), alpha(n - 1), l(n), mu(n), z(n);

    for (int i = 0; i < n - 1; ++i) {
        h[i] = x[i + 1] - x[i];
    }

    for (int i = 1; i < n - 1; ++i) {
        alpha[i] = (3 / h[i] * (y[i + 1] - y[i])) - (3 / h[i - 1] * (y[i] - y[i - 1]));
    }

    l[0] = 1;
    mu[0] = 0;
    z[0] = 0;

    for (int i = 1; i < n - 1; ++i) {
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    l[n - 1] = 1;
    z[n - 1] = 0;
    splines[n - 1].c = 0;

    for (int j = n - 2; j >= 0; --j) {
        splines[j].c = z[j] - mu[j] * splines[j + 1].c;
        splines[j].b = (y[j + 1] - y[j]) / h[j] - h[j] * (splines[j + 1].c + 2 * splines[j].c) / 3;
        splines[j].d = (splines[j + 1].c - splines[j].c) / (3 * h[j]);
        splines[j].a = y[j];
    }
}

// Функция для вычисления значения сплайна в точке x
double splineValue(const vector<Spline>& splines, double x) {
    int n = splines.size();
    Spline s;

    // Находим нужный сплайн для заданного x
    for (int i = 0; i < n - 1; ++i) {
        if (x >= splines[i].x && x <= splines[i + 1].x) {
            s = splines[i];
            break;
        }
    }

    double dx = x - s.x;
    return s.a + s.b * dx + s.c * dx * dx + s.d * dx * dx * dx;
}

int main() {
    vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    vector<double> y = {1.0, 0.86603, 0.5, 0.0, -0.5};
    int n = x.size();

    vector<Spline> splines(n);

    for (int i = 0; i < n; ++i) {
        splines[i].x = x[i];
    }

    cubicSpline(x, y, splines);

    double x_star = 1.5;
    double result = splineValue(splines, x_star);

    cout << "Value of the cubic spline at x* = " << x_star << " is: " << fixed << setprecision(5) << result << endl;

    return 0;
}
