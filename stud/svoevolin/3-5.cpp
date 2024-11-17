#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <functional>
#include <algorithm>
#include "matrix.h"

using namespace std;


double integrate_rectangles(function <double(double)> f, double x0, double x1, double h)
{
    double res = 0;
    while (x0 < x1)
    {
        double x = x0 + h;
        res += f((x + x0) / 2) * h;
        x0 = x;
    }
    return res;
}

double integrate_trapezoids(function <double(double)> f, double x0, double x1, double h)
{
    double res = 0;
    while (x0 < x1)
    {
        double x = x0 + h;
        res += (f(x0) + f(x)) * h;
        x0 = x;
    }
    return res / 2;
}

double integrate_simpson(function <double(double)> f, double x0, double x1, double h)
{
    double res = 0;
    while (x0 < x1)
    {
        double x = x0 + 2 * h;
        double xm = x0 + h;
        res += (f(x0) + 4 * f(xm) + f(x)) * h;
        x0 = x;
    }
    return res / 3;
}

// rectangles -> p = 2
// trapezoids -> p = 2
// simpson    -> p = 4
double method_runge(double i1, double i2, double h1, double h2, double p)
{
    double k = h2 / h1;
    return i1 + (i1 - i2) / (pow(k, p) - 1);
}

double f2(double x)
{
    return sqrt(x) / (4 + 3 * x);
}

int main()
{
    setlocale(LC_ALL, "Rus");
    ofstream fout("answer3-5.txt");
    fout.precision(5);
    fout << fixed;

    double h1 = 1, h2 = 0.5;
    fout << "Метод прямоугольников:\n";
    double i1 = integrate_rectangles(f2, 1, 5, h1);
    double i2 = integrate_rectangles(f2, 1, 5, h2);
    fout << "при h1: " << i1;
    fout << "\nпри h2: " << i2;
    fout << "\nуточнение Рунге-Ромберга: " << method_runge(i1, i2, h1, h2, 2);
    fout << "\nМетод трапеций:\n";
    i1 = integrate_trapezoids(f2, 1, 5, h1);
    i2 = integrate_trapezoids(f2, 1, 5, h2);
    fout << "при h1: " << i1;
    fout << "\nпри h2: " << i2;
    fout << "\nуточнение Рунге-Ромберга: " << method_runge(i1, i2, h1, h2, 2);
    fout << "\nМетод Симпсона:\n";
    i1 = integrate_simpson(f2, 1, 5, h1);
    i2 = integrate_simpson(f2, 1, 5, h2);
    fout << "при h1: " << i1;
    fout << "\nпри h2: " << i2;
    fout << "\nуточнение Рунге-Ромберга: " << method_runge(i1, i2, h1, h2, 4);
}