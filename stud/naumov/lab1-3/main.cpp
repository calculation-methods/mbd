#include <iostream>
#include <vector>
#include <cmath>

//using namespace std;

std::vector<double> solve_simple_iteration(std::vector<std::vector<double>>& matrix, std::vector<double>& vector_b, double precision, int &iterations, int max_iterations=1000) {
    int n = matrix.size();
    std::vector<double> x(n, 0.0);
    while (true) {
        iterations++;
        std::vector<double> x_new(n, 0.0);
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    sum += matrix[i][j] * x[j];
                }
            }
            x_new[i] = (vector_b[i] - sum) / matrix[i][i];
        }
        double max_diff = 0.0;
        for (int i = 0; i < n; i++) {
            max_diff = std::max(max_diff, std::fabs(x[i] - x_new[i]));
        }
        if (max_diff < precision) {
            break;
        }
        x = x_new;
        if (iterations > max_iterations) {
            std::cout << "Warning: The method of simple iterations did not converge." << std::endl;
            break;
        }
    }
    return x;
}

std::vector<double> SolveUsingGaussSeidel(std::vector<std::vector<double>>& matrix, std::vector<double>& vector_b, double precision, int& iterations, int max_iterations = 1000) {
    int n = matrix.size();
    std::vector<double> x(n, 0.0);
    std::vector<double> x_new(n, 0.0);
    while (true) {
        iterations++;
        for (int i = 0; i < n; i++) {
            double sum1 = 0.0;
            double sum2 = 0.0;
            for (int j = 0; j < i; j++) {
                sum1 += matrix[i][j] * x_new[j];
            }
            for (int j = i + 1; j < n; j++) {
                sum2 += matrix[i][j] * x[j];
            }
            x_new[i] = (vector_b[i] - sum1 - sum2) / matrix[i][i];
        }
        double max_diff = 0.0;
        for (int i = 0; i < n; i++) {
            max_diff = std::max(max_diff, std::fabs(x[i] - x_new[i]));
        }
        if (max_diff < precision) {
            break;
        }
        x = x_new;
        if (iterations > max_iterations) {
            break;
        }
    }
    return x;
}

bool check(std::vector<std::vector<double>> a, std::vector<double> x, std::vector<double> b) {
    double eps = 0.00001;
    for (int i = 0; i < a.size(); i++) {
        double row = 0.0;
        for (int j = 0; j < a[i].size(); j++){
            row += a[i][j] * x[j];
        }
        if (std::fabs(row - b[i]) > eps) {
            return false;
        }
    }
    return true;
}

int main() {
    std::vector<std::vector<double>> A = {{21, -6, -9, -4}, {-6, 20, -4, 2}, {-2, -7, -20, 3}, {4, 9, 6, 24}};
    std::vector<double> b = {127, -144, 236, -5};
    double eps = 0.000000001;
    int iteration_first = 0;
    int iteration_second = 0;

    std::vector<double> x_iter = solve_simple_iteration(A, b, eps, iteration_first);
    std::cout << "Iteration method - X: ";
    for (double xi : x_iter) {
        std::cout << xi << " ";
    }
    std::cout << std::endl;
    std::cout << "Number of iterations: " << iteration_first << std::endl;

    std::vector<double> x_seidel = SolveUsingGaussSeidel(A, b, eps, iteration_second);
    std::cout << "Seidel method - X: ";
    for (double xi : x_seidel) {
        std::cout << xi << " ";
    }
    std::cout << std::endl;
    std::cout << "Number of iterations: " << iteration_second << std::endl;

    std::cout << "Iteration method correct? - " << check(A, x_iter, b) << std::endl;
    std::cout << "Seidel method correct? - " << check(A, x_seidel, b) << std::endl;
    return 0;
}
