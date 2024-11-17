#include <iostream>
#include <vector>
#include <cmath>

double lagrange_interpolation(const std::vector<double>& x, const std::vector<double>& y, double x_star) {
    double result = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double term = y[i];
        for (size_t j = 0; j < x.size(); ++j) {
            if (j != i) {
                term *= (x_star - x[j]) / (x[i] - x[j]);
            }
        }
        result += term;
    }
    return result;
}

double diff(const std::vector<double>& x, const std::vector<double>& y, size_t n) {
    if (n == 0) {
        return y[0];
    } else {
        return (diff(x, y, n - 1) - diff(x, y, n - 1)) / (x[n] - x[0]);
    }
}

double newton_interpolation(const std::vector<double>& x, const std::vector<double>& y, double x_star) {
    double result = y[0];
    double term = 1.0;
    for (size_t i = 1; i < x.size(); ++i) {
        term *= (x_star - x[i - 1]);
        result += diff(x, y, i) * term;
    }
    return result;
}

int main() {
    std::vector<double> x = {0.1, 0.5, 0.9, 1.3};
    std::vector<double> y;
    for (double xi : x) {
        y.push_back(std::log(xi) + xi);
    }
    double x_star = 0.8;

    double interpolated_value = lagrange_interpolation(x, y, x_star);
    std::cout << "Interpolated value at x_star = " << interpolated_value << std::endl;

    double true_value = std::log(x_star) + x_star;
    double error = std::abs(true_value - interpolated_value);
    std::cout << "Real value = " << true_value << " Error at x_star = " << error << std::endl;

    x = {0.1, 0.5, 1.1, 1.3};
    y.clear();
    for (double xi : x) {
        y.push_back(std::log(xi) + xi);
    }
    interpolated_value = newton_interpolation(x, y, x_star);
    std::cout << "Interpolated value at x_star = " << interpolated_value << std::endl;

    true_value = std::log(x_star) + x_star;
    error = std::abs(true_value - interpolated_value);
    std::cout << "Real value = " << true_value << " Error at x_star = " << error << std::endl;

    return 0;
}
