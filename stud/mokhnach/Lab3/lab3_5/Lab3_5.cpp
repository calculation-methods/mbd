#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

double f(double x) {
    return (x * x * sqrt(36 - x * x));
}

double RungeRombergMethod(double F1, double h1, double F2,  double h2, double k) {
    return (F1 + (F1 - F2) / (pow(h2 / h1, k) - 1));
}


double RectangleMethod(double a, double b, double step) {
    double res = 0;
    double x_prev = a;
    for (double cur = a + step; cur <= b; cur += step) {
        res += step * f((x_prev + cur) / 2);
        x_prev = cur;
    }
    return res;
}

double SimpsonMethod(double a, double b, double step) {
    double res = 0;
    step = step/2;
    for (double cur = a + 2 * step; cur <= b; cur += 2 * step) {
        res += step * (f(cur - 2 * step) + 4 * f(cur - step) + f(cur));
    }
    return (1.0 / 3 * res);
}

double TrapezeMethod(double a, double b, double step) {
    double res = 0;
    double x_prev = a;
    for (double cur = a + step; cur <= b; cur += step) {
        res += step * (f(cur) + f(x_prev));
        x_prev = cur;
    }
    return 0.5 * res;
}

int main() {
    double X_0 = 0;
    double X_k = 2;
    double h1 = 0.5, h2 = 0.25;
    ofstream fout("answer.txt");

    fout << "Rectangle method:\n";
    double F_h1 = RectangleMethod(X_0, X_k, h1);
    double F_h2 = RectangleMethod(X_0, X_k, h2);
    double F = RungeRombergMethod(F_h1, F_h2, h1, h2, 10);
    fout << "F = " << F_h1 << "; h = " << h1 << "; error = " << abs(F - F_h1) << "\n";
    fout << "F = " << F_h2 << "; h = " << h2 << "; error = " << abs(F - F_h2) << "\n";

    fout << "\nTrapeze method:\n";
    F_h1 = TrapezeMethod(X_0, X_k, h1);
    F_h2 = TrapezeMethod(X_0, X_k, h2);
    F = RungeRombergMethod(F_h1, h1, F_h2,  h2, 10);
    fout << "F = " << F_h1 << "; h = " << h1 << "; error = " << abs(F - F_h1) << "\n";
    fout << "F = " << F_h2 << "; h = " << h2 << "; error = " << abs(F - F_h2) << "\n ";
    
    fout << "\nSimpson method:\n";
    F_h1 = SimpsonMethod(X_0, X_k, h1);
    F_h2 = SimpsonMethod(X_0, X_k, h2);
    F = RungeRombergMethod(F_h1, h1, F_h2,  h2, 10);
    fout << "F = " << SimpsonMethod(X_0, X_k, h1) << "; h = " << h1 << " error = " << abs(F - F_h1) << "\n";
    fout << "F = " << SimpsonMethod(X_0, X_k, h2) << "; h = " << h2 << " error = " << abs(F - F_h2) << "\n ";

    return 0;
}