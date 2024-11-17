#include <iostream>
#include <vector>

std::vector<double> run_through_method(std::vector<std::vector<double>> matrix, std::vector<double> b, int n) {
    std::vector<double> P(n, 0.0);
    std::vector<double> Q(n, 0.0);

    P[0] = -matrix[0][1] / matrix[0][0];
    Q[0] = b[0] / matrix[0][0];

    for (int i = 1; i < n - 1; ++i) {
        double denominator = matrix[i][i] + matrix[i][i - 1] * P[i - 1];
        P[i] = -matrix[i][i + 1] / denominator;
        Q[i] = (b[i] - matrix[i][i - 1] * Q[i - 1]) / denominator;
    }

    std::vector<double> x(n, 0.0);
    double denominator = matrix[n - 1][n - 1] + matrix[n - 1][n - 2] * P[n - 2];
    Q[n - 1] = (b[n - 1] - matrix[n - 1][n - 2] * Q[n - 2]) / denominator;
    x[n - 1] = Q[n - 1];

    for (int i = n - 2; i >= 0; --i) {
        x[i] = P[i] * x[i + 1] + Q[i];
    }

    return x;
}

void print_result(std::vector<double> res, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << res[i] << " ";
    }
}

int main() {
    std::vector<std::vector<double>> matrix = {{18.0, -9.0, 0.0, 0.0, 0.0},
                                               {2.0, -9.0, -4.0, 0.0, 0.0},
                                               {0.0, -9.0, 21.0, -8.0, 0.0},
                                               {0.0, 0.0, -4.0, -10.0, 5.0},
                                               {0.0, 0.0, 0.0, 7.0, 12.0}};
    std::vector<double> b = {-81.0, 71.0, -39.0, 64.0, 3.0};

    int n = 5;
    std::vector<double> result = run_through_method(matrix, b, n);
    print_result(result, n);
    return 0;
}
