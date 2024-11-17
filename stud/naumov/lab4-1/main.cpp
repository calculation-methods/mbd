#include <iostream>
#include <vector>
#include <cmath>


void eulerMethod(double h, double x0, double y10, double y20, double x_end, std::vector<double>& x_vals, std::vector<double>& y1_vals, std::vector<double>& y2_vals) {
    double x = x0;
    double y1 = y10;
    double y2 = y20;

    while (x <= x_end) {
        x_vals.push_back(x);
        y1_vals.push_back(y1);
        y2_vals.push_back(y2);

        double y1_new = y1 + h * y2;
        double y2_new = y2 + h * ((2 * x * y2 - 2 * y1) / (x * x - 1));

        y1 = y1_new;
        y2 = y2_new;
        x += h;
    }
}

void rungeKuttaMethod(double h, double x0, double y10, double y20, double x_end, std::vector<double>& x_vals, std::vector<double>& y1_vals, std::vector<double>& y2_vals) {
    double x = x0;
    double y1 = y10;
    double y2 = y20;

    while (x <= x_end) {
        x_vals.push_back(x);
        y1_vals.push_back(y1);
        y2_vals.push_back(y2);

        double k1_1 = h * y2;
        double k1_2 = h * ((2 * x * y2 - 2 * y1) / (x * x - 1));

        double k2_1 = h * (y2 + 0.5 * k1_2);
        double k2_2 = h * ((2 * (x + 0.5 * h) * (y2 + 0.5 * k1_2) - 2 * (y1 + 0.5 * k1_1)) / ((x + 0.5 * h) * (x + 0.5 * h) - 1));

        double k3_1 = h * (y2 + 0.5 * k2_2);
        double k3_2 = h * ((2 * (x + 0.5 * h) * (y2 + 0.5 * k2_2) - 2 * (y1 + 0.5 * k2_1)) / ((x + 0.5 * h) * (x + 0.5 * h) - 1));

        double k4_1 = h * (y2 + k3_2);
        double k4_2 = h * ((2 * (x + h) * (y2 + k3_2) - 2 * (y1 + k3_1)) / ((x + h) * (x + h) - 1));

        y1 += (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1) / 6;
        y2 += (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2) / 6;
        x += h;
    }
}

void adamsMethod(double h, double x0, double y10, double y20, double x_end, std::vector<double>& x_vals, std::vector<double>& y1_vals, std::vector<double>& y2_vals) {
    double x = x0;
    double y1 = y10;
    double y2 = y20;

    std::vector<double> f1, f2;

    // Используем метод Рунге-Кутты для начальных четырех шагов
    for (int i = 0; i < 4; i++) {
        x_vals.push_back(x);
        y1_vals.push_back(y1);
        y2_vals.push_back(y2);

        double k1_1 = h * y2;
        double k1_2 = h * ((2 * x * y2 - 2 * y1) / (x * x - 1));

        double k2_1 = h * (y2 + 0.5 * k1_2);
        double k2_2 = h * ((2 * (x + 0.5 * h) * (y2 + 0.5 * k1_2) - 2 * (y1 + 0.5 * k1_1)) / ((x + 0.5 * h) * (x + 0.5 * h) - 1));

        double k3_1 = h * (y2 + 0.5 * k2_2);
        double k3_2 = h * ((2 * (x + 0.5 * h) * (y2 + 0.5 * k2_2) - 2 * (y1 + 0.5 * k2_1)) / ((x + 0.5 * h) * (x + 0.5 * h) - 1));

        double k4_1 = h * (y2 + k3_2);
        double k4_2 = h * ((2 * (x + h) * (y2 + k3_2) - 2 * (y1 + k3_1)) / ((x + h) * (x + h) - 1));

        y1 += (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1) / 6;
        y2 += (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2) / 6;
        x += h;

        f1.push_back(y2);
        f2.push_back((2 * x * y2 - 2 * y1) / (x * x - 1));
    }

    while (x <= x_end) {
        x_vals.push_back(x);
        y1_vals.push_back(y1);
        y2_vals.push_back(y2);

        double y1_new = y1 + h / 24 * (55 * f1.back() - 59 * f1[f1.size() - 2] + 37 * f1[f1.size() - 3] - 9 * f1[f1.size() - 4]);
        double y2_new = y2 + h / 24 * (55 * f2.back() - 59 * f2[f2.size() - 2] + 37 * f2[f2.size() - 3] - 9 * f2[f2.size() - 4]);

        f1.push_back(y2_new);
        f2.push_back((2 * x * y2_new - 2 * y1_new) / (x * x - 1));

        y1 = y1_new;
        y2 = y2_new;
        x += h;
    }
}

double exactSolution(double x) {
    return x + 1 + exp(x);
}

int main() {
    double h = 0.1;
    double x0 = 2.0;
    double y10 = 7.0;
    double y20 = 5.0;
    double x_end = 3.0;
    std::vector<double> x_vals, y1_vals, y2_vals;
    std::vector<double> x_vals_half, y1_vals_half, y2_vals_half;

    std::cout << "Euler method: " << std::endl;
    eulerMethod(h, x0, y10, y20, x_end, x_vals, y1_vals, y2_vals);
    for (size_t i = 0; i < x_vals.size(); i++) {
        double y_exact = exactSolution(x_vals[i]);
        std::cout << "x: " << x_vals[i] << ", my solution: " << y1_vals[i] << ", exact solution: " << y_exact << ", error: " << fabs(y1_vals[i] - y_exact) << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Euler method error: " << std::endl;
    eulerMethod(h / 2, x0, y10, y20, x_end, x_vals_half, y1_vals_half, y2_vals_half);
    double error = fabs(y1_vals_half[y1_vals_half.size() - 1] - y1_vals[y1_vals.size() - 1]) / (pow(2, 4) - 1);
    std::cout << "Exact solution in x = " << x_end << ": " << exactSolution(x_end) << std::endl;
    std::cout << "My solution in x = " << x_end << ": " << y1_vals[y1_vals.size() - 1] << std::endl;
    std::cout << "Error: " << error << std::endl;
    std::cout << std::endl;
    x_vals.clear();
    y1_vals.clear();
    y2_vals.clear();
    x_vals_half.clear();
    y1_vals_half.clear();
    y2_vals_half.clear();

    std::cout << "Runge-Kutta method: " << std::endl;
    rungeKuttaMethod(h, x0, y10, y20, x_end, x_vals, y1_vals, y2_vals);
    for (size_t i = 0; i < x_vals.size(); i++) {
        double y_exact = exactSolution(x_vals[i]);
        std::cout << "x: " << x_vals[i] << ", my solution: " << y1_vals[i] << ", exact solution: " << y_exact << ", error: " << fabs(y1_vals[i] - y_exact) << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Runge-Kutta method error: " << std::endl;
    rungeKuttaMethod(h / 2, x0, y10, y20, x_end, x_vals_half, y1_vals_half, y2_vals_half);
    error = fabs(y1_vals_half[y1_vals_half.size() - 1] - y1_vals[y1_vals.size() - 1]) / (pow(2, 4) - 1);
    std::cout << "Exact solution in x = " << x_end << ": " << exactSolution(x_end) << std::endl;
    std::cout << "My solution in x = " << x_end << ": " << y1_vals[y1_vals.size() - 1] << std::endl;
    std::cout << "Error: " << error << std::endl;
    std::cout << std::endl;
    x_vals.clear();
    y1_vals.clear();
    y2_vals.clear();
    x_vals_half.clear();
    y1_vals_half.clear();
    y2_vals_half.clear();

    std::cout << "Adams method: " << std::endl;
    adamsMethod(h, x0, y10, y20, x_end, x_vals, y1_vals, y2_vals);
    for (size_t i = 0; i < x_vals.size(); i++) {
        double y_exact = exactSolution(x_vals[i]);
        std::cout << "x: " << x_vals[i] << ", my solution: " << y1_vals[i] << ", exact solution: " << y_exact << ", error: " << fabs(y1_vals[i] - y_exact) << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Adams method error: " << std::endl;
    adamsMethod(h / 2, x0, y10, y20, x_end, x_vals_half, y1_vals_half, y2_vals_half);
    error = fabs(y1_vals_half[y1_vals_half.size() - 1] - y1_vals[y1_vals.size() - 1]) / (pow(2, 4) - 1);
    std::cout << "Exact solution in x = " << x_end << ": " << exactSolution(x_end) << std::endl;
    std::cout << "My solution in x = " << x_end << ": " << y1_vals[y1_vals.size() - 1] << std::endl;
    std::cout << "Error: " << error << std::endl;
    std::cout << std::endl;
    x_vals.clear();
    y1_vals.clear();
    y2_vals.clear();
    x_vals_half.clear();
    y1_vals_half.clear();
    y2_vals_half.clear();

    return 0;
}
