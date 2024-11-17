#include <iostream>
#include <cmath>

double f1(double x1, double x2) {
    return 2 * x1 - std::cos(x2);
}

double f2(double x1, double x2) {
    return 2 * x2 - std::exp(x1);
}

double df1_dx1(double x2) {
    return 2;
}

double df1_dx2(double x1) {
    return std::sin(x1);
}

double df2_dx1(double x1) {
    return -std::exp(x1);
}

double df2_dx2() {
    return 2;
}

std::pair<double, double> simple_iteration(double accuracy, double x1, double x2, int &iters) {
    while (iters < 1000) {
        double new_x1 = std::cos(x2) / 2.0;
        double new_x2 = std::exp(x1) / 2.0;

        if (std::fabs(new_x1 - x1) < accuracy && std::fabs(new_x2 - x2) < accuracy) {
            return {new_x1, new_x2};
        }

        x1 = new_x1;
        x2 = new_x2;
        iters++;
    }

    return {x1, x2};
}

// Метод Ньютона
std::pair<double, double> newton_method(double accuracy, double x1, double x2, int &iters) {
    while (iters < 1000) {
        double det = df1_dx1(x2) * df2_dx2() - df1_dx2(x1) * df2_dx1(x1);
        double dx1 = (f1(x1, x2) * df2_dx2() - f2(x1, x2) * df1_dx2(x1)) / det;
        double dx2 = (f2(x1, x2) * df1_dx1(x2) - f1(x1, x2) * df2_dx1(x1)) / det;

        x1 -= dx1;
        x2 -= dx2;

        if (std::fabs(dx1) < accuracy && std::fabs(dx2) < accuracy) {
            return {x1, x2};
        }
        iters++;
    }

    return {x1, x2};
}

int main() {
    double accuracy = 0.001;
    double x1 = 0.5, x2 = 0.5;
    int iters = 0;
    auto simple_solution = simple_iteration(accuracy, x1, x2, iters);
    std::cout << "Simple Iteration Method Solution:\n";
    std::cout << "x1 = " << simple_solution.first << ", x2 = " << simple_solution.second << ", iterations = " << iters << std::endl;
    std::cout << std::endl;

    iters = 0;
    auto newton_solution = newton_method(accuracy, x1, x2, iters);
    std::cout << "Newton's Method Solution:\n";
    std::cout << "x1 = " << newton_solution.first << ", x2 = " << newton_solution.second << ", iterations = " << iters << std::endl;
    return 0;
}
