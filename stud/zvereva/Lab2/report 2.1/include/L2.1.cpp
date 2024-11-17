#include <iostream>
#include <cmath>

const double EPSILON = 0.001;

double func(double x) {
    return sin(x) - 2 * pow(x, 2) + 0.5;
}

double dfunc(double x) {
    return cos(x) - 4 * x;
}

double phi(double x) {
    return sqrt((sin(x) + 0.5) / 2);
}

int main() {
    double x0 = 0.4;
    while (abs(func(x0)) > EPSILON) {
        x0 = phi(x0);
    }

    std::cout << "The method of simple iterations" << std::endl;
    std::cout << x0 << std::endl;

    x0 = 0.5;
    double x1 = x0 - func(x0) / dfunc(x0);

    while (abs(x1 - x0) > EPSILON) {
        x0 = x1;
        x1 = x1 - func(x0) / dfunc(x0);
    }

    std::cout << "Newton's method" << std::endl;
    std::cout << x0 << std::endl;

    return 0;
}
