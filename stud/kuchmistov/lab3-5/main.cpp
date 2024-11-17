#include <iostream>
#include <cmath>

using namespace std;

double func(double x) {
    return 1.0 / (pow(x, 4) + 16);
}

double rectangle_method(double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += func(a + i * h);
    }
    return h * sum;
}

double trapezoidal_method(double a, double b, int n) {
    double h = (b - a) / n;
    double sum = (func(a) + func(b)) / 2.0;
    for (int i = 1; i < n; ++i) {
        sum += func(a + i * h);
    }
    return h * sum;
}

double simpson_method(double a, double b, int n) {
    if (n % 2 != 0) {
        cout << "Number of intervals must be even for Simpson's method." << endl;
        return 0;
    }
    double h = (b - a) / n;
    double sum = func(a) + func(b);
    for (int i = 1; i < n; ++i) {
        if (i % 2 == 0)
            sum += 2 * func(a + i * h);
        else
            sum += 4 * func(a + i * h);
    }
    return h * sum / 3.0;
}

double runge_romberg(double I_h, double I_2h, int p) {
    return (I_h - I_2h) / (pow(2, p) - 1);
}

int main() {
    double X_0 = 0.0;
    double X_1 = 2.0;
    double h_1 = 0.5;
    double h_2 = 0.25;

    double I_rectangle_h1 = rectangle_method(X_0, X_1, (X_1 - X_0) / h_1);
    double I_rectangle_h2 = rectangle_method(X_0, X_1, (X_1 - X_0) / h_2);

    double I_trapezoidal_h1 = trapezoidal_method(X_0, X_1, (X_1 - X_0) / h_1);
    double I_trapezoidal_h2 = trapezoidal_method(X_0, X_1, (X_1 - X_0) / h_2);

    double I_simpson_h1 = simpson_method(X_0, X_1, (X_1 - X_0) / h_1);
    double I_simpson_h2 = simpson_method(X_0, X_1, (X_1 - X_0) / h_2);

    double error_rectangle = runge_romberg(I_rectangle_h1, I_rectangle_h2, 2);
    double error_trapezoidal = runge_romberg(I_trapezoidal_h1, I_trapezoidal_h2, 2);
    double error_simpson = runge_romberg(I_simpson_h1, I_simpson_h2, 4);

    cout << "Rectangle method (h=0.5): " << I_rectangle_h1 << endl;
    cout << "Rectangle method (h=0.25): " << I_rectangle_h2 << endl;
    cout << "Error estimate (rectangles): " << error_rectangle << endl << endl;

    cout << "Trapezoidal method (h=0.5): " << I_trapezoidal_h1 << endl;
    cout << "Trapezoidal method (h=0.25): " << I_trapezoidal_h2 << endl;
    cout << "Error estimate (trapezoids): " << error_trapezoidal << endl << endl;

    cout << "Simpson's method (h=0.5): " << I_simpson_h1 << endl;
    cout << "Simpson's method (h=0.25): " << I_simpson_h2 << endl;
    cout << "Error estimate (Simpson): " << error_simpson << endl;

    return 0;
}
