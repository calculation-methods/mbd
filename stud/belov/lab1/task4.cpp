#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

std::vector<std::vector<double>> find_transport_matrix(const std::vector<std::vector<double>>& matrix) {
    std::vector<std::vector<double>> transport_matrix = matrix;
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = i + 1; j < matrix.size(); ++j) {
            std::swap(transport_matrix[i][j], transport_matrix[j][i]);
        }
    }
    return transport_matrix;
}

std::vector<std::vector<double>> make_e_matrix(int size) {
    std::vector<std::vector<double>> e_matrix(size, std::vector<double>(size, 0));
    for (int i = 0; i < size; ++i) {
        e_matrix[i][i] = 1.0;
    }
    return e_matrix;
}

double scholar_multiply(const std::vector<double>& vector_1, const std::vector<double>& vector_2) {
    double result = 0;
    for (size_t i = 0; i < vector_1.size(); ++i) {
        result += vector_1[i] * vector_2[i];
    }
    return result;
}


std::vector<std::vector<double>> multiply_matrix(const std::vector<std::vector<double>>& matrix_1, const std::vector<std::vector<double>>& matrix_2) {
    size_t n = matrix_1.size();
    size_t m = matrix_1[0].size();
    size_t p = matrix_2[0].size();
    if (m != matrix_2.size()) {
        throw std::invalid_argument("Matrix dimensions do not match.");
    }

    std::vector<std::vector<double>> result(n, std::vector<double>(p, 0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) {
            for (size_t k = 0; k < m; ++k) {
                result[i][j] += matrix_1[i][k] * matrix_2[k][j];
            }
        }
    }
    return result;
}


std::vector<int> find_max_non_diagonal(const std::vector<std::vector<double>>& matrix) {
    double max_val = matrix[0][1];
    std::vector<int> max_indices = {0, 1};
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = i + 1; j < matrix[i].size(); ++j) {
            if (std::abs(matrix[i][j]) > std::abs(max_val)) {
                max_val = matrix[i][j];
                max_indices = {static_cast<int>(i), static_cast<int>(j)};
            }
        }
    }
    return max_indices;
}

double find_phi(const std::vector<std::vector<double>>& matrix, int i, int j) {
    if (matrix[i][i] == matrix[j][j]) {
        return M_PI / 4;
    }
    return 0.5 * std::atan((2 * matrix[i][j]) / (matrix[i][i] - matrix[j][j]));
}

std::vector<std::vector<double>> make_rotation_matrix(double angle, size_t size, int i_, int j_) {
    std::vector<std::vector<double>> res = make_e_matrix(size);
    res[i_][j_] = -std::sin(angle);
    res[j_][i_] = std::sin(angle);
    res[i_][i_] = res[j_][j_] = std::cos(angle);
    return res;
}

double find_lim(const std::vector<std::vector<double>>& matrix) {
    double res = 0;
    for (size_t i = 0; i < matrix.size() - 1; ++i) {
        for (size_t j = i + 1; j < matrix[i].size(); ++j) {
            res += matrix[i][j] * matrix[i][j];
        }
    }
    return std::sqrt(res);
}



std::pair<std::vector<std::vector<double>>, std::vector<double>> rotation_solution(const std::vector<std::vector<double>>& matrix, double epsilon) {
    std::vector<std::vector<double>> matrix_a = matrix;
    std::vector<std::vector<double>> matrix_res_u = make_e_matrix(matrix.size());
    while (std::abs(find_lim(matrix_a)) > epsilon) {
        std::vector<int> res_ij = find_max_non_diagonal(matrix_a);
        double phi = find_phi(matrix_a, res_ij[0], res_ij[1]);
        std::vector<std::vector<double>> matrix_u = make_rotation_matrix(phi, matrix_a.size(), res_ij[0], res_ij[1]);
        matrix_res_u = multiply_matrix(matrix_res_u, matrix_u);
        matrix_a = multiply_matrix(multiply_matrix(find_transport_matrix(matrix_u), matrix_a), matrix_u);
    }
    std::vector<std::vector<double>> result_vector;
    std::vector<double> lambdas(matrix_a.size());
    for (size_t i = 0; i < matrix_a.size(); ++i) {
        result_vector.push_back({matrix_res_u[i]});
        lambdas[i] = matrix_a[i][i];
    }

    return std::make_pair(matrix_res_u, lambdas);
}

int main() {
    std::vector<std::vector<double>> matrix = {
            {7.0, 3.0, -1.0},
            {3.0, -7.0, -8.0},
            {-1.0, -8.0, -2.0}
    };


    auto rotation_solution_result = rotation_solution(matrix, 0.01);
    auto matrix_res_u = rotation_solution_result.first;
    auto lambdas = rotation_solution_result.second;

    std::cout << "Solution for Rotation method:\n";
    for (const auto& row : matrix_res_u) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Lambdas\n";
    for (auto const& var: lambdas) {
        std::cout << var << " ";
    }

    return 0;
}

