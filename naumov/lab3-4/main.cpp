#include <iostream>
#include <vector>

double first_derivative(const std::vector<double>& x, const std::vector<double>& y, double x_star) {
    int idx = -1;
    for (int i = 0; i < x.size(); ++i) {
        if (x[i] <= x_star)
            idx = i;
        else
            break;
    }
    if (idx == -1 || idx == x.size() - 1) {
        std::cerr << "x_star is outside the range of the table." << std::endl;
        return 0.0;
    }
    double h = x[1] - x[0];
    double derivative = (y[idx + 1] - y[idx - 1]) / (2 * h);

    return derivative;
}

double second_derivative(const std::vector<double>& x, const std::vector<double>& y, double x_star) {
    int idx = -1;
    for (int i = 0; i < x.size(); ++i) {
        if (x[i] <= x_star)
            idx = i;
        else
            break;
    }

    if (idx == -1 || idx == x.size() - 1) {
        std::cerr << "x_star is outside the range of the table." << std::endl;
        return 0.0;
    }

    double h = x[1] - x[0];
    double derivative = (y[idx + 1] - 2 * y[idx] + y[idx - 1]) / (h * h);

    return derivative;
}

int main() {
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {0.0, 2.0, 3.4142, 4.7321, 6.0};
    double x_star = 2.0;

    double first_deriv = first_derivative(x, y, x_star);
    std::cout << "First derivative at x_star = " << first_deriv << std::endl;

    double second_deriv = second_derivative(x, y, x_star);
    std::cout << "Second derivative at x_star = " << second_deriv << std::endl;

    return 0;
}
