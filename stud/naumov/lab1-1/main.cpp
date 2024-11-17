#include <iostream>
#include <vector>
#include <tuple>

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>> lu_decomposition_with_pivoting(const std::vector<std::vector<double>>& matrix, int n) {
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> U(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> P(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; i++)
        P[i][i] = 1.0;

    for (int i = 0; i < n; i++) {
        double max_val = 0.0;
        for (int j = 0; j < n; j++)
            max_val = std::max(max_val, std::abs(matrix[i][j]));

        if (max_val == 0.0)
            return std::make_tuple(std::vector<std::vector<double>>(), std::vector<std::vector<double>>(), std::vector<std::vector<double>>()); // Матрица вырожденная

        for (int j = 0; j < n; j++) {
            if (i <= j) {
                U[i][j] = matrix[i][j];
                for (int k = 0; k < i; ++k)
                    U[i][j] -= L[i][k] * U[k][j];
            }
            if (i > j) {
                L[i][j] = matrix[i][j];
                for (int k = 0; k < j; ++k)
                    L[i][j] -= L[i][k] * U[k][j];
                L[i][j] /= U[j][j];
            }
        }
    }

    return std::make_tuple(L, U, P);
}

std::vector<double> solve_lu_decomposition(const std::vector<std::vector<double>>& L,
                                           const std::vector<std::vector<double>>& U,
                                           const std::vector<std::vector<double>>& P,
                                           const std::vector<double>& b, int n) {
    std::vector<double> y(n, 0.0);
    std::vector<double> x(n, 0.0);

    // Решаем Ly = Pb
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
        for (int j = 0; j < n; j++)
            y[i] += P[i][j] * b[j];
        for (int j = 0; j < i; j++)
            y[i] -= L[i][j] * y[j];
    }

    // Решаем Ux = y
    for (int i = n - 1; i >= 0; --i) {
        x[i] = y[i];
        for (int j = i + 1; j < n; j++)
            x[i] -= U[i][j] * x[j];
        x[i] /= U[i][i];
    }

    return x;
}

double determinant_from_lu(const std::vector<std::vector<double>>& U, const std::vector<std::vector<double>>& P, int n) {
    double det_U = 1.0;
    for (int i = 0; i < n; i++)
        det_U *= U[i][i];
    double det_P = 1.0;
    for (int i = 0; i < n; i++)
        det_P *= P[i][i];
    return det_U * det_P;
}

std::vector<std::vector<double>> inverse_from_lu(const std::vector<std::vector<double>>& L,
                                                 const std::vector<std::vector<double>>& U,
                                                 const std::vector<std::vector<double>>& P,
                                                 int n) {
    std::vector<std::vector<double>> inv(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        std::vector<double> b(n, 0.0);
        b[i] = 1.0;
        std::vector<double> x = solve_lu_decomposition(L, U, P, b, n);
        for (int j = 0; j < n; j++)
            inv[j][i] = x[j];
    }
    return inv;
}

void print_results(std::vector<double> x, double det, std::vector<std::vector<double>> inv, int n) {

    std:: cout << "Solution X = { ";
    for (int i = 0; i < n; i++) {
        std::cout << x[i] << " ";
    }
    std:: cout << "}" << std::endl;

    std:: cout << "Matrix Inv = {" << std::endl;
    for (int i =0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << inv[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std:: cout << "}" << std::endl;

    std::cout << "Det = " << det << std::endl;
}

int main() {
    std::vector<std::vector<double>> A = {{-5.0, -1.0, -3.0, -1.0},
                                          {-2.0, 0.0, 8.0, -4.0},
                                          {-7.0, -2.0, 2.0, -2.0},
                                          {2.0, -4.0, -4.0, 4.0}};
    std::vector<double> b = {18.0, -12.0, 6.0, -12.0};

    int n = 4;

    std::vector<std::vector<double>> L, U, P;
    std::tie(L, U, P) = lu_decomposition_with_pivoting(A, n);
    if (L.empty()) {
        std::cout << "Solution does not exist." << std::endl;
        return 1;
    }

    std::vector<double> x = solve_lu_decomposition(L, U, P, b, n);
    double det = determinant_from_lu(U, P, n);
    std::vector<std::vector<double>> inv = inverse_from_lu(L, U, P, n);
    print_results(x, det, inv, n);
    return 0;
}
