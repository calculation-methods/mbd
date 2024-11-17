#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>

class NotANeededMatrix : public std::exception {
public:
    const char* what() const noexcept override {
        return "Sorry this matrix doesn't fit the requirements!\n";
    }
};

double find_det(const std::vector<std::vector<double>>& matrix) {
    double det = 1;
    for (size_t i = 0; i < matrix.size(); ++i) {
        det *= matrix[i][i];
    }
    return det;
}

void check_coefficients(double a = 0, double b = 0, double c = 0) {
    return;
}

std::vector<double> tridiagonal_solution(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vector_b) {
    check_coefficients(matrix[0][0], matrix[0][1]);
    std::vector<double> vector_alphas = {-matrix[0][1] / matrix[0][0]};
    std::vector<double> vector_betas = {vector_b[0] / matrix[0][0]};

    for (size_t i = 1; i < matrix.size() - 1; ++i) {
        check_coefficients(matrix[i][i - 1], matrix[i][i], matrix[i][i + 1]);
        double y_i = matrix[i][i] + matrix[i][i - 1] * vector_alphas[i - 1];
        double a_i = -matrix[i][i + 1] / y_i;
        double b_i = (vector_b[i] - matrix[i][i - 1] * vector_betas[i - 1]) / y_i;
        vector_alphas.push_back(a_i);
        vector_betas.push_back(b_i);
    }

    check_coefficients(matrix.back()[matrix.back().size() - 2], matrix.back().back());
    double y_n = (matrix.back().back() + matrix.back()[matrix.back().size() - 2] * vector_alphas.back());
    std::vector<double> vector_x = {round((vector_b.back() - matrix.back()[matrix.back().size() - 2] * vector_betas.back()) / y_n)};

    for (int i = static_cast<int>(matrix.size()) - 2; i >= 0; --i) {
        vector_x.insert(vector_x.begin(), round(vector_alphas[i] * vector_x[0] + vector_betas[i]));
    }

    return vector_x;
}

int main() {

    std::vector<std::vector<double>> matrix = {
            {10.0, 5.0, 0, 0, 0},
            {3.0, 10.0, -2.0, 0.0, 0},
            {0, 2, -9, -5, 0},
            {0, 0, 5, 16, -4},
            {0, 0, 0, -8, 16}
    };

    std::vector<double> vector = {-120, -91, 5, -74, -56};
    find_det(matrix);
    auto solution = tridiagonal_solution(matrix, vector);
    std::cout << "The solution for matrix:\n";
    for (const auto& row : matrix) {
        for (double val : row) {
            std::cout << val << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "Is this vector:\n";
    for (double val : solution) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

