#include <iostream>
#include <cmath>

double f(double x) {
    return x * exp(x) + x * x - 1;
}

double f_derivative(double x) {
    return exp(x) + x * exp(x) + 2 * x;
}


double g(double x) {
    return x - (x * std::exp(x) + x * x - 1) / (std::exp(x) + 2 * x);
}

double Iteration(double initialGuess, double accuracy, int &iterations) {
    double x = initialGuess;
    while (iterations < 1000) {
        double x1 = g(x);
        if (std::fabs(x1 - x) < accuracy) {
            return x1;
        }
        x = x1;
        iterations++;
    }
    return x;
}

double Newton(double initialGuess, double accuracy, int &iterations) {
    double x = initialGuess;
    while (fabs(f(x)) > accuracy && iterations < 1000) {
        x = x - f(x) / f_derivative(x);
        iterations++;
    }
    return x;
}

int main() {
    double initialGuess = 0.6;
    double accuracy = 0.001;
    int iterations = 0;

    double answer = Iteration(initialGuess, accuracy, iterations);
    std::cout << "Iteration Method: Answer = " << answer << ", iterations = " << iterations << std::endl;

    iterations = 0;
    answer = Newton(initialGuess, accuracy, iterations);
    std::cout << "Newton Method: Answer = " << answer << ", iterations = " << iterations << std::endl;

    return 0;
}
