#include <iostream>
#include <cmath>
#include <vector>

double getEpsilon(std::vector<std::vector<double>>& A) {
    double res = 0;
    for (int i = 0; i < A.size(); ++i) {
        for (int j = i + 1; j < A[i].size(); ++j) {
            res += A[i][j] * A[i][j];
        }
    }
    return std::sqrt(res);
}

double multiplyMatricesCell(const std::vector<std::vector<double>>& firstMatrix,
    const std::vector<std::vector<double>>& secondMatrix, int row, int col) {
    double cell = 0;
    for (int i = 0; i < secondMatrix.size(); ++i) {
        cell += firstMatrix[row][i] * secondMatrix[i][col];
    }
    return cell;
}


std::vector<std::vector<double>> multiplyMatrices(const std::vector<std::vector<double>>& firstMatrix,
    const std::vector<std::vector<double>>& secondMatrix) {
    std::vector<std::vector<double>> result(firstMatrix.size(), std::vector<double>(secondMatrix[0].size(), 0.0));

    for (int row = 0; row < result.size(); ++row) {
        for (int col = 0; col < result[row].size(); ++col) {
            result[row][col] = multiplyMatricesCell(firstMatrix, secondMatrix, row, col);
        }
    }

    return result;
}


void printMatrix(const std::vector<std::vector<double>>& matrix) {
    for (const auto& row : matrix) {
        std::cout << "|";
        for (const auto& d : row) {
            std::cout << "\t" << d << "\t";
        }
        std::cout << "|\n";
    }
}

std::vector<std::vector<double>> transport(const std::vector<std::vector<double>>& A) {
    int n = A.size();
    std::vector<std::vector<double>> transposeMatrix(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            transposeMatrix[j][i] = A[i][j];
        }
    }
    return transposeMatrix;
}

std::vector<double> findMax(const std::vector<std::vector<double>>& A) {
    int maxi = 0;
    int maxj = 0;
    double max_val = 0;
    for (int i = 0; i < A.size(); ++i) {
        for (int j = i + 1; j < A[i].size(); ++j) {
            if (abs(A[i][j]) > max_val) {
                max_val = abs(A[i][j]);
                maxi = i;
                maxj = j;
            }
        }
    }
    return { max_val, static_cast<double>(maxi), static_cast<double>(maxj) };
}

std::vector<std::vector<double>> getU(const std::vector<std::vector<double>>& A, int ai, int aj) {
    int n = A.size();
    double phi;
    if (A[ai][ai] == A[aj][aj])
        phi = 3.1415926535 / 4;
    else
        phi = 0.5 * std::atan(2 * A[ai][aj] / (A[ai][ai] - A[aj][aj]));

    std::vector<std::vector<double>> U(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j)
                U[i][j] = 1;
            else
                U[i][j] = 0;
        }
    }
    U[ai][ai] = std::cos(phi);
    U[aj][aj] = std::cos(phi);
    U[ai][aj] = -std::sin(phi);
    U[aj][ai] = std::sin(phi);

    return U;
}


const double EPSILON = 0.1;

int main() {
    std::vector<std::vector<double>> A = {
        { -7, -6, 8 },
        { -6, 3, -7 },
        { 8, -7, 4 }
    };

    std::vector<std::vector<double>> Ak = A;

    std::vector<std::vector<std::vector<double>>> ListU;

    double epsilon = getEpsilon(A);
    int count = 0;
    while (epsilon > EPSILON && count < 10) {
        std::vector<double> a = findMax(Ak);
        std::vector<std::vector<double>> U = getU(Ak, (int)a[1], (int)a[2]);
        ListU.push_back(U);
        Ak = multiplyMatrices(multiplyMatrices(transport(U), Ak), U);
        epsilon = getEpsilon(Ak);
        std::cout << epsilon << std::endl;
        ++count;
    }

    std::cout << "Eigenvalues\n[";
    for (int i = 0; i < Ak.size(); ++i) {
        std::cout << " " << Ak[i][i] << " ";
    }
    std::cout << "]\n";

    std::cout << "\n================================\n";
    std::vector<std::vector<double>> V = multiplyMatrices(ListU[0], ListU[1]);
    for (int i = 2; i < ListU.size(); ++i) {
        V = multiplyMatrices(V, ListU[i]);
    }

    printMatrix(V);

    return 0;
}