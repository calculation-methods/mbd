#include <iostream>
#include <cmath>

double y(double x) {
    return x * x / (pow(x, 4) + 256);
}

double rectangle_method(double x0, double x1, double h) {
    double integral = 0.0;
    for (double x = x0; x < x1; x += h) {
        integral += y(x) * h;
    }
    return integral;
}

double trapezoid_method(double x0, double x1, double h) {
    double integral = 0.0;
    for (double x = x0; x < x1; x += h) {
        integral += (y(x) + y(x + h)) * h / 2.0;
    }
    return integral;
}

double simpson_method(double x0, double x1, double h) {
    double integral = 0.0;
    for (double x = x0; x < x1; x += 2 * h) {
        integral += (y(x) + 4 * y(x + h) + y(x + 2 * h)) * h / 3.0;
    }
    return integral;
}

double runge_romberg_method(double I1, double I2, double p) {
    return (I2 - I1) / (pow(2, p) - 1);
}

int main() {
    double x0 = 0.0;
    double x1 = 2.0;
    double h1 = 0.5;
    double h2 = 0.25;

    double integral_rect_h1 = rectangle_method(x0, x1, h1);
    double integral_rect_h2 = rectangle_method(x0, x1, h2);

    double integral_trap_h1 = trapezoid_method(x0, x1, h1);
    double integral_trap_h2 = trapezoid_method(x0, x1, h2);

    double integral_simpson_h1 = simpson_method(x0, x1, h1);
    double integral_simpson_h2 = simpson_method(x0, x1, h2);

    double error_rect = runge_romberg_method(integral_rect_h1, integral_rect_h2, 2);
    double error_trap = runge_romberg_method(integral_trap_h1, integral_trap_h2, 2);
    double error_simpson = runge_romberg_method(integral_simpson_h1, integral_simpson_h2, 4);

    std::cout << "Rectangle method integral with h1: " << integral_rect_h1 << std::endl;
    std::cout << "Trapezoid method integral with h1: " << integral_trap_h1 << std::endl;
    std::cout << "Simpson method integral with h1: " << integral_simpson_h1 << std::endl;

    std::cout << "Rectangle method integral with h2: " << integral_rect_h2 << std::endl;
    std::cout << "Trapezoid method integral with h2: " << integral_trap_h2 << std::endl;
    std::cout << "Simpson method integral with h2: " << integral_simpson_h2 << std::endl;


    std::cout << "Estimated error for rectangle method: " << error_rect << std::endl;
    std::cout << "Estimated error for trapezoid method: " << error_trap << std::endl;
    std::cout << "Estimated error for Simpson method: " << error_simpson << std::endl;

    return 0;
}
