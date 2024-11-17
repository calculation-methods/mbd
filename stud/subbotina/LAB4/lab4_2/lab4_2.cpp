#include "solver.h"
#include <locale.h>

using namespace std;

double g(double x, double y, double z) {
    if (x == 0)
        return z;
    else
        return ((2 * x + 4) * z - 2 * y) / (x * x + 4 * x);
       
}

double f(double x, double y, double z) {
    (void)x;
    (void)y;
    return z;
}

double px(double x) {
        return -(2 * x + 4) / (x * x + 4 * x);
}

double qx(double x) {
        return 2 / (x * x + 4 * x);
}

double fx(double x) {
    (void)x;
    return 0.0;
}

using tddd = tuple<double, double, double>;

int main() {
    setlocale(0, "");
    cout.precision(6);
    cout << fixed;
    double h = 0.1 , eps = 0.00001;
    double a = 0, b = 2;
    double alpha = 0, beta = 1, y0 = 1;
    double delta = 1, gamma = -1, y1 = 3;

    shooting de_shooting(a, b, f, g, alpha, beta, y0, delta, gamma, y1);
    vector<tddd> sol_shooting = de_shooting.solve(h, eps);
    cout << "Метод стрельбы:" << endl;
    print_data(sol_shooting);
    cout << "Погрешность вычислений:" << endl;
    double shooting_err = runge_romberg(de_shooting.solve(h, eps), de_shooting.solve(h / 2, eps), 4);
    cout << shooting_err << endl;

    fin_dif de_fin_dif(a, b, px, qx, fx, alpha, beta, y0, delta, gamma, y1);
    vector<tddd> sol_fin_dif = de_fin_dif.solve(h);
    cout << "Конечно-разностный метод:" << endl;
    print_data(sol_fin_dif);
    cout << "Погрешность вычислений:" << endl;
    double fin_dif_err = runge_romberg(de_fin_dif.solve(h), de_fin_dif.solve(h / 2), 2);
    cout << fin_dif_err << endl;
}