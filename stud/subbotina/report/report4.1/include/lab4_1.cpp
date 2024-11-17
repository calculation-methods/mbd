#include <cmath>
#include <iostream>
#include <locale.h>
#include "desolve.h"

using namespace std;

double g(double x, double y, double z) {
    return (((x*x-2) * y) / (x * x));
}

double f(double x, double y, double z) {
    return (sin(x-1) + 1/x*cos(x-1));
}

using tddd = tuple<double, double, double>;

int main() {
    setlocale(0, "");
    cout.precision(6);
    cout << fixed;
    double l = 1, r = 2, y0 = 1, z0 = 0, h;
    cout << "Введите шаг сетки h: ";
    cin >> h;

    euler de_euler(l, r, f, g, y0, z0);
    vector<tddd> sol_euler = de_euler.solve(h);
    cout << "Метод Эйлера:" << endl;
    print_data(sol_euler);
    cout << "Погрешность вычислений:" << endl;
    double euler_err = runge_romberg(de_euler.solve(h), de_euler.solve(h / 2), 1);
    cout << euler_err << endl;

    runge de_runge(l, r, f, g, y0, z0);
    vector<tddd> sol_runge = de_runge.solve(h);
    cout << "Метод Рунге-Кутты:" << endl;
    print_data(sol_runge);
    cout << "Погрешность вычислений:" << endl;
    double runge_err = runge_romberg(de_runge.solve(h), de_runge.solve(h / 2), 4);
    cout << runge_err << endl;

    adams de_adams(l, r, f, g, y0, z0);
    vector<tddd> sol_adams = de_adams.solve(h);
    cout << "Метод Адамса:" << endl;
    print_data(sol_adams);
    cout << "Погрешность вычислений:" << endl;
    double adams_err = runge_romberg(de_adams.solve(h), de_adams.solve(h / 2), 4);
    cout << adams_err << endl;

}