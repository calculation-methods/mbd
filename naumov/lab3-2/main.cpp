#include <iostream>
#include <vector>
#include <cmath>

void cubic_spline(const std::vector<double>& x, const std::vector<double>& f, double x_star) {
    size_t n = x.size();

    std::vector<double> h(n - 1);
    std::vector<double> alpha(n - 1);
    std::vector<double> l(n);
    std::vector<double> mu(n - 1);
    std::vector<double> z(n);

    for (size_t i = 0; i < n - 1; ++i) {
        h[i] = x[i + 1] - x[i];
    }

    for (size_t i = 1; i < n - 1; ++i) {
        alpha[i] = 3 * (f[i + 1] - f[i]) / h[i] - 3 * (f[i] - f[i - 1]) / h[i - 1];
    }

    l[0] = 1;
    mu[0] = 0;
    z[0] = 0;

    for (size_t i = 1; i < n - 1; ++i) {
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    l[n - 1] = 1;
    z[n - 1] = 0;
    std::vector<double> c(n);
    std::vector<double> b(n - 1);
    std::vector<double> d(n - 1);

    for (int j = n - 2; j >= 0; --j) {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (f[j + 1] - f[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
        d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
    }

    size_t interval_index = 0;
    for (size_t i = 0; i < n - 1; ++i) {
        if (x[i] <= x_star && x_star <= x[i + 1]) {
            interval_index = i;
            break;
        }
    }

    double A = f[interval_index];
    double B = b[interval_index];
    double C = c[interval_index];
    double D = d[interval_index];
    double interpolated_value = A + B * (x_star - x[interval_index]) + C * pow((x_star - x[interval_index]), 2) + D * pow((x_star - x[interval_index]), 3);

    std::cout << "Interpolated value at x_star = " << interpolated_value << std::endl;
}

int main() {
    std::vector<double> x = {0.1, 0.5, 0.9, 1.3, 1.7};
    std::vector<double> f = {-2.2026, -0.19315, 0.79464, 1.5624, 2.2306};
    double x_star = 0.8;

    cubic_spline(x, f, x_star);

    return 0;
}
