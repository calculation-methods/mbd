#include <iostream>
#include <cmath>
#include <algorithm>
#include <locale.h>


using namespace std;
int iterCount = 0;

double f(double x) {
    return pow(10, x) - 5 * x - 2;
}

double der(double x) {
    return log(10) * pow(10, x) - 5;
}

double der2(double x) {
    return log(10) * log(10) * pow(10, x);
}

double phi(double x) {
    return log(5 * x + 2) / log(10);
}

double phi_der(double x) {
    return 5 / (log(10) * (5 * x + 2));
}

double iterations(double l, double r, double eps) {
    iterCount = 0;
    double x_k = (l + r) / 2;
    double dx = 1;
    double q = std::max(std::abs(phi_der(l)), std::abs(phi_der(r)));
    double eps_coeff = q / (1 - q);
    while (dx > eps) {
        double x_k1 = phi(x_k);
        dx = eps_coeff * std::abs(x_k1 - x_k);
        ++iterCount;
        x_k = x_k1;
    }
    return x_k;
}

double newton(double l, double r, double eps) {
    iterCount = 0;
    double x_k = l;
    if (f(l) * f(r) >= 0) {
        return 0;
    }
    if (f(x_k) * der2(x_k) <= 0) {
        x_k = r;
    }
    double dx = 1;
    while (dx > eps) {
        double x_k1 = x_k - f(x_k) / der(x_k);
        dx = std::abs(x_k1 - x_k);
        ++iterCount;
        x_k = x_k1;
    }
    return x_k;
}


int main() {
    setlocale(0, "");
    cout.precision(9);
    cout << fixed;
    double l = 1, r = 0.5, eps;
    cout << "Введите точность вычислений:";
    cin >> eps;
    double root;
    root = iterations(l, r, eps);
    cout << "Метод простой итерации:\n";
    cout << "Решение нелинейного уравнения:x_0 = " << root << '\n';
    cout << "Количество итераций: " << iterCount << '\n';
    root = newton(l, r, eps);
    cout << "Метод Ньютона:\n";
    cout << "Решение нелинейного уравнения:x_0 = " << root << '\n';
    cout << "Количество итераций: " << iterCount << '\n';
}