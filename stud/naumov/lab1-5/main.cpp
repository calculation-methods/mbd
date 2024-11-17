#include <iostream>
#include <vector>
#include <cmath>

void qr_decomposition(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& Q, std::vector<std::vector<double>>& R,double eps) {
    int n = A.size();
    int m = A[0].size();

    Q = A;
    R = std::vector<std::vector<double>>(m, std::vector<double>(m, 0.0));

    for (int j = 0; j < m; ++j) {
        // Compute the j-th column of Q and R
        for (int k = 0; k < j; ++k) {
            double dot_product = 0.0;
            for (int i = 0; i < n; ++i) {
                dot_product += Q[i][j] * Q[i][k];
            }
            for (int i = 0; i < n; ++i) {
                Q[i][j] -= dot_product * Q[i][k];
            }
        }

        double norm = 0.0;
        for (int i = 0; i < n; ++i) {
            norm += Q[i][j] * Q[i][j];
        }
        norm = sqrt(norm);

        for (int i = 0; i < n; ++i) {
            Q[i][j] /= norm;
            R[j][j] = norm;
        }

        for (int k = j + 1; k < m; ++k) {
            double dot_product = 0.0;
            for (int i = 0; i < n; ++i) {
                dot_product += Q[i][j] * A[i][k];
            }
            R[j][k] = dot_product;
        }
    }
}

std::vector<std::vector<double>> matrix_multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    int n = A.size();
    int m = B[0].size();
    int p = B.size();

    std::vector<std::vector<double>> result(n, std::vector<double>(m, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < p; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

std::vector<double> compute_eigenvalues(const std::vector<std::vector<double>>& A, int iterations, double eps) {
    int n = A.size();

    std::vector<std::vector<double>> Ak = A;

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<std::vector<double>> Q, R;
        qr_decomposition(Ak, Q, R, eps);
        Ak = matrix_multiply(R, Q);
    }

    std::vector<double> eigenvalues(n);

    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = Ak[i][i];
    }

    return eigenvalues;
}


int main() {
    std::vector<std::vector<double>> A = {{1, 2, 5}, {-8, 0, -6}, {7, -9, -7}};

    double epsilon = 1e-6;

    std::vector<std::vector<double>> Q;
    std::vector<std::vector<double>> R;

    qr_decomposition(A, Q, R, epsilon);

    std::vector<double> eigenvalues = compute_eigenvalues(A, 50, epsilon);

    std::cout << "Eigenvalues:" << std::endl;
    for (double eigenvalue : eigenvalues) {
        std::cout << eigenvalue << " ";
    }
    std::cout << std::endl;

    return 0;
}
