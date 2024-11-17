#include <iostream>
#include <vector>
#include <cmath>


void solve_system(std::vector<std::vector<double>>& A, std::vector<double>& b, std::vector<double>& x) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        int pivot = i;
        for (int j = i + 1; j < n; ++j) {
            if (abs(A[j][i]) > abs(A[pivot][i])) {
                pivot = j;
            }
        }
        std::swap(A[i], A[pivot]);
        std::swap(b[i], b[pivot]);
        for (int j = i + 1; j < n; ++j) {
            double factor = A[j][i] / A[i][i];
            for (int k = i; k < n; ++k) {
                A[j][k] -= factor * A[i][k];
            }
            b[j] -= factor * b[i];
        }
    }
    x.assign(n, 0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0;
        for (int j = i + 1; j < n; ++j) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    }
}

std::vector<double> least_squares_polynomial(const std::vector<double>& x, const std::vector<double>& f, int degree) {
    int n = x.size();
    std::vector<std::vector<double>> A(degree + 1, std::vector<double>(degree + 1, 0));
    std::vector<double> b(degree + 1, 0);
    for (int i = 0; i <= degree; ++i) {
        for (int j = 0; j <= degree; ++j) {
            for (int k = 0; k < n; ++k) {
                A[i][j] += pow(x[k], i + j);
            }
        }
        for (int k = 0; k < n; ++k) {
            b[i] += pow(x[k], i) * f[k];
        }
    }
    std::vector<double> coefficients;
    solve_system(A, b, coefficients);
    return coefficients;
}

double evaluate_polynomial(const std::vector<double>& coefficients, double x) {
    double result = 0;
    for (size_t i = 0; i < coefficients.size(); ++i) {
        result += coefficients[i] * pow(x, i);
    }
    return result;
}

double sum_of_squared_errors(const std::vector<double>& x, const std::vector<double>& f, const std::vector<double>& coefficients) {
    double sum = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        double error = f[i] - evaluate_polynomial(coefficients, x[i]);
        sum += error * error;
    }
    return sum;
}

int main() {
    std::vector<double> x = {0.1, 0.5, 0.9, 1.3, 1.7, 2.1};
    std::vector<double> f = {-2.2026, -0.19315, 0.79464, 1.5624, 2.2306, 2.8419};

    std::vector<double> coefficients_degree_1 = least_squares_polynomial(x, f, 1);
    double sum_of_squared_errors_degree_1 = sum_of_squared_errors(x, f, coefficients_degree_1);
    std::cout << "Coefficients for degree 1 polynomial: ";
    for (size_t i = 0; i < coefficients_degree_1.size(); ++i) {
        std::cout << coefficients_degree_1[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Sum of squared errors for degree 1 polynomial: " << sum_of_squared_errors_degree_1 << std::endl;

    std::vector<double> coefficients_degree_2 = least_squares_polynomial(x, f, 2);
    double sum_of_squared_errors_degree_2 = sum_of_squared_errors(x, f, coefficients_degree_2);
    std::cout << "Coefficients for degree 2 polynomial: ";
    for (size_t i = 0; i < coefficients_degree_2.size(); ++i) {
        std::cout << coefficients_degree_2[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Sum of squared errors for degree 2 polynomial: " << sum_of_squared_errors_degree_2 << std::endl;

    return 0;
}
