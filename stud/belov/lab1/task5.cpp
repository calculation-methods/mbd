#include <iostream>
#include <utility>
#include <vector>
#include <cmath>
#include <complex>

double row_col_mul(const std::vector<double>& row, const std::vector<double>& col) {
    double result = 0;
    for (size_t i = 0; i < row.size(); ++i) {
        result += row[i] * col[i];
    }
    return result;
}

std::pair<std::complex<double>, std::complex<double>> solve_equation(double a, double b, double c) {
    std::complex<double> delta = std::sqrt(std::complex<double>(b * b - 4 * a * c, 0));
    std::complex<double> root1 = (b + delta) / (2 * a);
    std::complex<double> root2 = (b - delta) / (2 * a);
    return std::make_pair(root1, root2);
}

std::vector<std::vector<double>> col_row_mul(const std::vector<double>& col, const std::vector<double>& row) {
    std::vector<std::vector<double>> result(col.size(), std::vector<double>(row.size(), 0));
    for (size_t i = 0; i < col.size(); ++i) {
        for (size_t j = 0; j < row.size(); ++j) {
            result[i][j] = col[i] * row[j];
        }
    }
    return result;
}

std::vector<std::vector<double>> matrix_matrix_mul(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2) {
    std::vector<std::vector<double>> result(matrix1.size(), std::vector<double>(matrix2[0].size(), 0));
    for (size_t i = 0; i < matrix1.size(); ++i) {
        for (size_t j = 0; j < matrix2[0].size(); ++j) {
            for (size_t k = 0; k < matrix1[0].size(); ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return result;
}

std::vector<std::vector<double>> get_e_matrix(size_t size) {
    std::vector<std::vector<double>> result(size, std::vector<double>(size, 0));
    for (size_t i = 0; i < size; ++i) {
        result[i][i] = 1;
    }
    return result;
}

int sign(double element) {
    if (element > 0) {
        return 1;
    } else if (element < 0) {
        return -1;
    }
    return 0;
}

double vector_second_norm_2(const std::vector<double>& vector) {
    double result = 0;
    for (double val : vector) {
        result += val * val;
    }
    return std::sqrt(result);
}

std::vector<std::vector<double>> get_h(const std::vector<double>& vector) {
    double lower = row_col_mul(vector, vector);
    auto upper = col_row_mul(vector, vector);
    auto E = get_e_matrix(upper.size());
    std::vector<std::vector<double>> second(upper.size(), std::vector<double>(upper.size(), 0));
    for (size_t i = 0; i < upper.size(); ++i) {
        for (size_t j = 0; j < upper.size(); ++j) {
            second[i][j] = -2 / lower * upper[i][j];
        }
    }
    std::vector<std::vector<double>> result(upper.size(), std::vector<double>(upper.size(), 0));
    for (size_t i = 0; i < upper.size(); ++i) {
        for (size_t j = 0; j < upper.size(); ++j) {
            result[i][j] = E[i][j] + second[i][j];
        }
    }
    return result;
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> gen_qr(std::vector<std::vector<double>> matrix) {
    auto A = matrix;
    std::vector<std::vector<double>> Q(matrix.size(), std::vector<double>(matrix.size(), 0));
    std::vector<std::vector<std::vector<double>>> H_all;
    std::vector<double> vector(matrix.size());
    for (size_t i = 0; i < matrix.size() - 1; ++i) {
        vector.clear();
        std::vector<double> row;
        for (const auto& col : A) {
            row.push_back(col[i]);
        }
        vector.insert(vector.end(), i, 0);
        vector.push_back(A[i][i] + sign(A[i][i]) * vector_second_norm_2(row));
        vector.insert(vector.end(), row.begin() + i + 1, row.end());
        auto H = get_h(vector);
        A = matrix_matrix_mul(H, A);
        H_all.push_back(H);
    }
    Q = H_all[0];
    for (size_t i = 1; i < H_all.size(); ++i) {
        Q = matrix_matrix_mul(Q, H_all[i]);
    }
    return std::make_pair(Q, A);
}

std::pair<double, std::pair<std::complex<double>, std::complex<double>>> qr_solve(std::vector<std::vector<double>> matrix, double eps) {
    auto A = std::move(matrix);
    while (sqrt(A[1][0] * A[1][0] + A[2][0] * A[2][0]) > eps) {
        auto qr = gen_qr(A);
        A = matrix_matrix_mul(qr.second, qr.first);
    }
    auto roots = solve_equation(A[1][1], A[2][2], A[1][2] * A[2][1]);
    return std::make_pair(A[0][0], roots);
}

int main() {
    // Example usage
    std::vector<std::vector<double>> matrix = {{-6, -4, 0},
                                               {-7, 6, -7},
                                               {-2, -6, -7}};
    double epsilon = 0.0001;
    auto result = qr_solve(matrix, epsilon);
    std::cout << "Eigenvalue: " << result.first << std::endl;
    std::cout << "Eigenvector: ";
    std::cout << result.second.first << " " << result.second.second;
    std::cout << std::endl;
    return 0;
}
