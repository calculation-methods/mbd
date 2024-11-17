#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

using namespace std;

// Функция для вычисления значения y = x / (3x + 4)^2
double func(double x) {
    return x / pow((3 * x + 4), 2);
}

// Метод прямоугольников
double rectangle_method(double x0, double xk, double h) {
    double F = 0;
    double n = (int)((xk - x0) / h);
    n += 1;
    vector<double> x_values(n, 0);
    for (int i = 0; i < n; i++) {
        x_values[i] = x0 + h * i;
    }
    for (int i = 1; i < n; i++) {
        F += h * func((x_values[i] + x_values[i - 1]) / 2);
    }
    return F;
}

// Метод трапеций
double trapez_method(double x0, double xk, double h) {
    double F = 0;
    double n = (int)((xk - x0) / h);
    n += 1;
    vector<double> x_values(n, 0);
    for (int i = 0; i < n; i++) {
        x_values[i] = x0 + h * i;
    }
    for (int i = 1; i < n; i++) {
        F += (func(x_values[i]) + func(x_values[i - 1])) / 2 * h;
    }
    return F;
}

// Метод Симпсона
double simps_method(double x0, double xk, double h) {
    int n = (int)((xk - x0) / h);
    double F = 0;
    for (int i = 0; i < n; i++) {
        double x1 = x0 + i * h;
        double x2 = x0 + (i + 1) * h;
        double x3 = x0 + (i + 0.5) * h;
        F += (h / 6) * (func(x1) + 4 * func(x3) + func(x2));
    }
    return F;
}

// Оценка погрешности методом Рунге-Ромберга-Ричардсона
vector<double> runge_romb_rich(double x0, double xk, double h, int p) {
    vector<double> results(3, 0);
    results[0] = rectangle_method(x0, xk, h / 2) + (rectangle_method(x0, xk, h / 2) - rectangle_method(x0, xk, h)) / (pow(2, p) - 1);
    results[1] = trapez_method(x0, xk, h / 2) + (trapez_method(x0, xk, h / 2) - trapez_method(x0, xk, h)) / (pow(2, p) - 1);
    results[2] = simps_method(x0, xk, h / 2) + (simps_method(x0, xk, h / 2) - simps_method(x0, xk, h)) / (pow(2, p) - 1);
    return results;
}

int main() {
    ofstream fout("answer5.txt");

    double x0 = 0, xk = 4, h1 = 1.0, h2 = 0.5;

    fout << "Rectangle with h = 1\n";
    double rect1 = rectangle_method(x0, xk, h1);
    fout << rect1;
    fout << "\nRectangle with h = 0.5\n";
    double rect2 = rectangle_method(x0, xk, h2);
    fout << rect2;

    fout << "\n\nTrapez with h = 1\n";
    double trap1 = trapez_method(x0, xk, h1);
    fout << trap1;
    fout << "\nTrapez with h = 0.5\n";
    double trap2 = trapez_method(x0, xk, h2);
    fout << trap2;

    fout << "\n\nSimpson with h = 1\n";
    double simp1 = simps_method(x0, xk, h1);
    fout << simp1;
    fout << "\nSimpson with h = 0.5\n";
    double simp2 = simps_method(x0, xk, h2);
    fout << simp2 << endl << endl;

    vector<double> RRR1 = runge_romb_rich(x0, xk, h1, 1);
    vector<double> RRR2 = runge_romb_rich(x0, xk, h2, 1);

    fout << "with Runge-Romberg-Richardson method" << endl;

    fout << "Rectangle with h = 1\n";
    fout << RRR1[0];
    fout << "\nEstimate:" << fabs(rect1 - RRR1[0]);

    fout << "\nRectangle with h = 0.5\n";
    fout << RRR2[0];
    fout << "\nEstimate:" << fabs(rect2 - RRR2[0]);

    fout << "\n\nTrapez with h = 1\n";
    fout << RRR1[1];
    fout << "\nEstimate:" << fabs(trap1 - RRR1[1]);

    fout << "\nTrapez with h = 0.5\n";
    fout << RRR2[1];
    fout << "\nEstimate:" << fabs(trap2 - RRR2[1]);

    fout << "\n\nSimpson with h = 1\n";
    fout << RRR1[2];
    fout << "\nEstimate:" << fabs(simp1 - RRR1[2]);

    fout << "\nSimpson with h = 0.5\n";
    fout << RRR2[2];
    fout << "\nEstimate:" << fabs(simp2 - RRR2[2]);

    fout.close();

    return 0;
}
