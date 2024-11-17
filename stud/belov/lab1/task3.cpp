#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>

std::vector<std::vector<double>> deepcopy(const std::vector<std::vector<double>>& matrix) {
    std::vector<std::vector<double>> result(matrix.size(), std::vector<double>(matrix[0].size()));
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            result[i][j] = matrix[i][j];
        }
    }
    return result;
}



std::pair<std::vector<std::vector<double>>, std::vector<double>> read_matrix_from_file(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> matrix;
    std::vector<double> vector;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            std::istringstream iss(line);
            std::vector<double> row;
            double value;
            while (iss >> value) {
                row.push_back(value);
            }
            if (vector.empty()) {
                vector = row;
            } else {
                matrix.push_back(row);
            }
        }
    }
    return std::make_pair(matrix, vector);
}

double matrix_norm1(const std::vector<std::vector<double>>& matrix) {
    double max_sum = 0.0;
    for (const auto& row : matrix) {
        double row_sum = 0.0;
        for (double val : row) {
            row_sum += std::abs(val);
        }
        if (row_sum > max_sum) {
            max_sum = row_sum;
        }
    }
    return max_sum;
}


double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

std::vector<double> subtract_vectors(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

double norm(const std::vector<double>& vector) {
    double max_val = vector[0];
    for (size_t i = 1; i < vector.size(); ++i) {
        if (vector[i] > max_val) {
            max_val = vector[i];
        }
    }
    return max_val;
}

std::pair<std::vector<std::vector<double>>, std::vector<double>> normal_view(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vector) {
    std::vector<std::vector<double>> res = matrix;
    std::vector<double> res_v = vector;
    for (size_t i = 0; i < res.size(); ++i) {
        double delim = res[i][i];
        for (size_t j = 0; j < res.size(); ++j) {
            res[i][j] /= -delim;
        }
        res_v[i] /= delim;
        res[i][i] = 0;
    }
    return std::make_pair(res, res_v);
}

std::vector<double> sum_vectors(const std::vector<double>& vect1, const std::vector<double>& vect2) {
    std::vector<double> result(vect1.size());
    for (size_t i = 0; i < vect1.size(); ++i) {
        result[i] = vect1[i] + vect2[i];
    }
    return result;
}

std::vector<double> prod_matrix(const std::vector<std::vector<double>>& a, const std::vector<double>& vector) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < a[i].size(); ++j) {
            sum += a[i][j] * vector[j];
        }
        result[i] = sum;
    }
    return result;
}

std::vector<double> gauss_seidel(const std::vector<std::vector<double>>& a, const std::vector<double>& b, double epsilon) {
    auto [a_norm, b_norm] = normal_view(a, b);
    double alpha_norm = matrix_norm1(a_norm);
    std::vector<double> x_start(a_norm.size(), 0);
    std::vector<double> x_new = b_norm;

    while (true) {
        if (alpha_norm / (1 - alpha_norm) * norm(subtract_vectors(x_new, x_start)) <= epsilon) {
            break;
        }
        x_start = x_new;
        for (size_t j = 0; j < a_norm.size(); ++j) {
            double x_res = 0;
            for (size_t l = 0; l < a_norm.size(); ++l) {
                x_res += x_new[l] * a_norm[j][l];
            }
            x_res += b_norm[j];
            x_new[j] = x_res;
        }
    }
    return x_new;
}

std::vector<double> simple_iteration(const std::vector<std::vector<double>>& a, const std::vector<double>& b, double epsilon) {
    auto [a_norm, b_norm] = normal_view(a, b);
    double alpha_norm = matrix_norm1(a_norm);
    std::vector<double> x_start(a_norm.size(), 0);
    size_t max_iters = 100000;
    std::vector<double> x_new = b_norm;

    for (size_t j = 0; j < max_iters; ++j) {
        if (alpha_norm / (1 - alpha_norm) * norm(subtract_vectors(x_new, x_start)) > epsilon) {
            x_start = x_new;
            x_new = sum_vectors(prod_matrix(a_norm, x_new), b_norm);
        } else {
            break;
        }
    }
    return x_new;
}
int main() {
    std::vector<std::vector<double>> matrix = {
            {24.0, 2.0, 4.0, -9.0},
            {-6.0, 27.0, -8.0, -6.0},
            {-4.0, 8.0, 19.0, 6.0},
            {4.0, 5.0, -3.0, -13.0}
    };

    // Создание вектора 1x4 и заполнение его значениями
    std::vector<double> vector = {-9.0, -76.0, -79.0, -70.0};
    double epsilon = 0.0001;
    auto result = simple_iteration(matrix, vector, epsilon);
    auto result2 = gauss_seidel(matrix, vector, epsilon);
    std::cout << "Simple Iterations\n";
    for (double element : result) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    std::cout << "Zeidel\n";
    for (double element : result2) {
        std::cout << element << " ";
    }

    std::cout << std::endl;
    return 0;
}
