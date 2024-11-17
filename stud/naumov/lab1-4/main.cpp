#include <iostream>
#include <vector>
#include <cmath>

constexpr double EPS = 0.01;


double rmsNonDiagonal(const std::vector<std::vector<double>>& matrix) {
    double sum = 0.0;
    int n = matrix.size();
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            sum += matrix[i][j] * matrix[i][j];
    return std::sqrt(sum);
}

std::pair<int, int> maxUpDiagonalElement(const std::vector<std::vector<double>>& A) {
    double maxElement = -std::numeric_limits<double>::max();
    std::pair<int, int> maxIndex = std::make_pair(0, 0);
    int size = A.size();
    for (int i = 0; i < size - 1; ++i)
        for (int j = i + 1; j < size; ++j)
            if (std::abs(A[i][j]) > maxElement) {
                maxElement = std::abs(A[i][j]);
                maxIndex = std::make_pair(i, j);
            }
    return maxIndex;
}

double Get_Phi(int max_i, int max_j, const std::vector<std::vector<double>>& A) {
    if(A[max_i][max_i] == A[max_j][max_j])
        return M_PI / 4;
    else
        return 0.5 * std::atan(2 * A[max_i][max_j] / (A[max_i][max_i] - A[max_j][max_j]));
}

std::vector<std::vector<double>> Transpose_Matrix(const std::vector<std::vector<double>>& A) {
    int rows = A.size();
    int cols = A[0].size();
    std::vector<std::vector<double>> result(cols, std::vector<double>(rows, 0));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result[j][i] = A[i][j];
    return result;
}

std::vector<std::vector<double>> Matrix_Multiplication(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    int n = A.size();
    int m = B[0].size();
    int p = B.size();
    std::vector<std::vector<double>> C(n, std::vector<double>(m, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < p; ++k)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

std::vector<std::vector<double>> Initialize_U(const std::vector<std::vector<double>>& A) {
    int im, jm;
    std::tie(im, jm) = maxUpDiagonalElement(A);
    double phi = Get_Phi(im, jm, A);

    int size = A.size();
    std::vector<std::vector<double>> U(size, std::vector<double>(size, 0));
    for (int i = 0; i < size; ++i)
        U[i][i] = 1;
    U[im][jm] = -std::sin(phi);
    U[jm][im] = std::sin(phi);
    U[im][im] = U[jm][jm] = std::cos(phi);
    return U;
}

void Jacobi_Eigenvalue(std::vector<std::vector<double>> A) {
    std::vector<std::vector<double>> V(A.size(), std::vector<double>(A.size(), 0));
    for(int i = 0; i < A.size(); ++i)
        V[i][i] = 1;

    while(rmsNonDiagonal(A) > EPS) {
        std::vector<std::vector<double>> U = Initialize_U(A);
        std::vector<std::vector<double>> U_t = Transpose_Matrix(U);
        A = Matrix_Multiplication(Matrix_Multiplication(U_t, A), U);
        V = Matrix_Multiplication(V, U);
    }

    std::vector<double> lambda(A.size());
    for(int i = 0; i < A.size(); ++i)
        lambda[i] = A[i][i];

    std::cout << "Eigenvalues:" << std::endl;
    for(int i = 0; i < lambda.size(); ++i)
        std::cout << "\t lambda " << i << " = " << lambda[i] << std::endl;
    std::cout << "Eigenvectors:" << std::endl;
    for(int j = 0; j < V.size(); ++j){
        std::cout << j << ":" << std::endl;
        for(int i = 0; i < V.size(); ++i)
            std::cout << "\t" << V[i][j] << std::endl;
    }
}

int main() {
    std::vector<std::vector<double>> A = {
            {8, -3, 9},
            {-3, 8, -2},
            {9, -2, -8}
    };
    Jacobi_Eigenvalue(A);
    return 0;
}
