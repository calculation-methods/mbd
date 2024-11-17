#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

double f(double x, double y, double yp) {
    return ((exp(x) + 1) * yp - 2 * y - exp(x) * y);
}

void rungeKutta(double h, double x0, double y0, double yp0, double xf, vector<double>& x_vals, vector<double>& y_vals) {
    double k1, k2, k3, k4;
    double y = y0, yp = yp0;
    for (double x = x0; x < xf; x += h) {
        x_vals.push_back(x);
        y_vals.push_back(y);
        k1 = h * yp;
        k2 = h * (yp + 0.5 * k1);
        k3 = h * (yp + 0.5 * k2);
        k4 = h * (yp + k3);
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
        yp = yp + (f(x, y, yp) + f(x + h, y + k1, yp + k1)) / 2;
    }
}

double shootingMethod(double h, double x0, double y0, double x_end, double y_end, double initial_guess) {
    double tolerance = 1e-6;
    double guess1 = initial_guess;
    double guess2 = initial_guess + 0.1;

    double f1, f2;

    while (true) {
        vector<double> x_vals, y_vals1, y_vals2;
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

void finiteDifferenceMethod(double h, double x0, double y0, double x_end, double y_end, vector<double>& x_vals, vector<double>& y_vals) {
    int n = (x_end - x0) / h;
    vector<double> a(n + 1), b(n + 1), c(n + 1), d(n + 1), y(n + 1);

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

int main() {
    double x0 = 0.0, xf = 1.0;
    double y0 = 0.0, yf = exp(1) - 1;
    double h = 0.1; // Step size
    double epsilon = 1e-6;

vector<double> exact_solution_x, exact_solution_y;
for (double x = x0; x <= xf; x += h) {
    exact_solution_x.push_back(x);
    exact_solution_y.push_back(exp(x) - 1);
}

    vector<double> shooting_solution_x, shooting_solution_y;
    double initial_guess = (yf - y0) / (xf - x0);
    double yp0 = shootingMethod(h, x0, y0, xf, yf, initial_guess);
    rungeKutta(h, x0, y0, yp0, xf, shooting_solution_x, shooting_solution_y);

    vector<double> fd_solution_x, fd_solution_y;
    finiteDifferenceMethod(h, x0, y0, xf, yf, fd_solution_x, fd_solution_y);

    cout << "x\tExact\tShooting\tFinite Difference\n";
    cout.precision(6);
    cout.setf(ios::fixed);
    for (int i = 0; i < exact_solution_x.size(); ++i) {
        cout << exact_solution_x[i] << "\t" << exact_solution_y[i] << "\t" << shooting_solution_y[i] << "\t\t" << fd_solution_y[i] << endl;
    }

double shooting_error = 0.0;
vector<double> shooting_solution_x_half, shooting_solution_y_half;
rungeKutta(h / 2, x0, y0, yp0, xf, shooting_solution_x_half, shooting_solution_y_half);
for (int i = 0; i < shooting_solution_x.size(); ++i) {
    double error = abs(shooting_solution_y_half[i] - shooting_solution_y[i]) / 15;
    if (error > shooting_error) {
        shooting_error = error;
    }
}

double fd_error = 0.0;
vector<double> fd_solution_x_half, fd_solution_y_half;
finiteDifferenceMethod(h / 2, x0, y0, xf, yf, fd_solution_x_half, fd_solution_y_half);
for (int i = 0; i < fd_solution_x.size(); ++i) {
    double error = abs(fd_solution_y_half[i] - fd_solution_y[i]) / 3;
    if (error > fd_error) {
        fd_error = error;
    }
}

cout << "\nRunge-Romberg Error (Shooting Method): " << shooting_error << endl;
cout << "Runge-Romberg Error (Finite Difference Method): " << fd_error << endl;

    return 0;
}
