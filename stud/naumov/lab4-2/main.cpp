#include <iostream>
#include <vector>
#include <cmath>


void rungeKutta(double h, double x0, double y0, double dy0, double x_end, std::vector<double>& x_vals, std::vector<double>& y_vals) {
    double x = x0;
    double y = y0;
    double dy = dy0;

    while (x <= x_end) {
        x_vals.push_back(x);
        y_vals.push_back(y);

        double k1 = h * dy;
        double l1 = h * (tan(x) * dy - 2 * y);
        double k2 = h * (dy + 0.5 * l1);
        double l2 = h * (tan(x + 0.5 * h) * (dy + 0.5 * l1) - 2 * (y + 0.5 * k1));
        double k3 = h * (dy + 0.5 * l2);
        double l3 = h * (tan(x + 0.5 * h) * (dy + 0.5 * l2) - 2 * (y + 0.5 * k2));
        double k4 = h * (dy + l3);
        double l4 = h * (tan(x + h) * (dy + l3) - 2 * (y + k3));

        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6;
        dy += (l1 + 2 * l2 + 2 * l3 + l4) / 6;
        x += h;
    }
}

double shootingMethod(double h, double x0, double y0, double x_end, double y_end, double initial_guess) {
    double tolerance = 1e-6;
    double guess1 = initial_guess;
    double guess2 = initial_guess + 0.1;

    double f1, f2;

    while (true) {
        std::vector<double> x_vals, y_vals1, y_vals2;
        rungeKutta(h, x0, y0, guess1, x_end, x_vals, y_vals1);
        rungeKutta(h, x0, y0, guess2, x_end, x_vals, y_vals2);

        f1 = y_vals1.back() - y_end;
        f2 = y_vals2.back() - y_end;

        if (fabs(f1) < tolerance) {
            return guess1;
        }
        if (fabs(f2) < tolerance) {
            return guess2;
        }

        double guess_new = guess1 - f1 * (guess2 - guess1) / (f2 - f1);
        guess1 = guess2;
        guess2 = guess_new;
    }
}

void finiteDifferenceMethod(double h, double x0, double y0, double x_end, double y_end, std::vector<double>& x_vals, std::vector<double>& y_vals) {
    int n = (x_end - x0) / h;
    std::vector<double> a(n + 1), b(n + 1), c(n + 1), d(n + 1), y(n + 1);

    for (int i = 0; i <= n; ++i) {
        x_vals.push_back(x0 + i * h);
    }

    a[0] = 0;
    b[0] = 1;
    c[0] = 0;
    d[0] = y0;

    for (int i = 1; i < n; ++i) {
        double x = x0 + i * h;
        a[i] = 1 / (h * h) - tan(x) / (2 * h);
        b[i] = -2 / (h * h) + 2;
        c[i] = 1 / (h * h) + tan(x) / (2 * h);
        d[i] = 0;
    }

    a[n] = 0;
    b[n] = 1;
    c[n] = 0;
    d[n] = y_end;

    for (int i = 1; i <= n; ++i) {
        double m = a[i] / b[i - 1];
        b[i] -= m * c[i - 1];
        d[i] -= m * d[i - 1];
    }

    y[n] = d[n] / b[n];
    for (int i = n - 1; i >= 0; --i) {
        y[i] = (d[i] - c[i] * y[i + 1]) / b[i];
    }

    for (int i = 0; i <= n; ++i) {
        y_vals.push_back(y[i]);
    }
}

double exactSolution(double x) {
    return sin(x) + 2 - sin(x) * log((1 + sin(x)) / (1 - sin(x)));
}

int main() {
    double h = 0.05;
    double x0 = 0.0;
    double y0 = 2.0;
    double x_end = M_PI / 6;
    double y_end = 2.5 - 0.5 * log(3.0);
    std::vector<double> x_vals, y_vals;
    std::vector<double> x_vals_half, y_vals_half;

    std::cout << "Shooting method: " << std::endl;
    double initial_guess = 0.0;
    double dy0 = shootingMethod(h, x0, y0, x_end, y_end, initial_guess);
    rungeKutta(h, x0, y0, dy0, x_end, x_vals, y_vals);
    for (size_t i = 0; i < x_vals.size(); i++) {
        double y_exact = exactSolution(x_vals[i]);
        std::cout << "x: " << x_vals[i] << ", my solution: " << y_vals[i] << ",| exact solution: " << y_exact << ",| error: " << fabs(y_vals[i] - y_exact) << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Shooting method error: " << std::endl;
    dy0 = shootingMethod(h/2, x0, y0, x_end, y_end, initial_guess);
    rungeKutta(h/2, x0, y0, dy0, x_end, x_vals_half, y_vals_half);
    double error = fabs(y_vals_half[y_vals_half.size() - 1] - y_vals[y_vals.size() - 1]) / (pow(2, 4) - 1);
    std::cout << "Exact solution in x = " << x_end << ": " << exactSolution(x_end) << std::endl;
    std::cout << "My solution in x = " << x_end << ": " << y_vals[y_vals.size() - 1] << std::endl;
    std::cout << "Error: " << error << std::endl;
    std::cout << std::endl;
    x_vals.clear();
    y_vals.clear();
    x_vals_half.clear();
    y_vals_half.clear();


    std::cout << "Finite Difference method: " << std::endl;
    finiteDifferenceMethod(h, x0, y0, x_end, y_end, x_vals, y_vals);
    for (size_t i = 0; i < x_vals.size(); i++) {
        double y_exact = exactSolution(x_vals[i]);
        std::cout << "x: " << x_vals[i] << ", my solution: " << y_vals[i] << ",| exact solution: " << y_exact << ",| error: " << fabs(y_vals[i] - y_exact) << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Finite Difference method error: " << std::endl;
    finiteDifferenceMethod(h/2, x0, y0, x_end, y_end, x_vals_half, y_vals_half);
    error = fabs(y_vals_half[y_vals_half.size() - 1] - y_vals[y_vals.size() - 1]) / (pow(2, 4) - 1);
    std::cout << "Exact solution in x = " << x_end << ": " << exactSolution(x_end) << std::endl;
    std::cout << "My solution in x = " << x_end << ": " << y_vals[y_vals.size() - 1] << std::endl;
    std::cout << "Error: " << error << std::endl;
    std::cout << std::endl;
    x_vals.clear();
    y_vals.clear();
    x_vals_half.clear();
    y_vals_half.clear();
    return 0;
}
