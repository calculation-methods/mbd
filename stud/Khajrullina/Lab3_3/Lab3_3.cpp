#include <iostream>
#include <vector>
#include <cmath>

std::vector<double> x = { 0.1,    0.5,    0.9,    1.3,    1.7,    2.1 };
std::vector<double> y = { 100.01,    4.25,    2.0446,    2.2817,    3.236,    4.6368 };

double pol(std::vector<double> a, double x) {
    double sum = 0;
    for (int i = 0; i < a.size(); ++i) {
        sum += pow(x, i) * a[i];
    }
    return sum;
}

double error(std::vector<double> a, std::vector<double> x, std::vector<double> y) {
    double sum = 0;
    for (int i = 0; i < x.size(); ++i) {
        sum += pow(pol(a, x[i]) - y[i], 2);
    }
    return sum;
}



std::vector<std::vector<std::vector<double>>> getLU(std::vector<std::vector<double>> A) {
    std::vector<std::vector<double>> U = A;

    std::vector<std::vector<double>> L(U.size(), std::vector<double>(U.size()));
    for (auto& l : L) std::fill(l.begin(), l.end(), 0.0);

    for (int k = 0; k < U.size(); ++k) {
        for (int i = k; i < U.size(); ++i) {
            L[i][k] = U[i][k] / U[k][k];
        }

        for (int i = k + 1; i < U.size(); ++i) {
            for (int j = k; j < U.size(); ++j) {
                U[i][j] = U[i][j] - L[i][k] * U[k][j];
            }
        }
    }

    return { L, U };
}

std::vector<double> solution(std::vector<std::vector<double>> A, std::vector<double> b) {
    int n = A.size();
    std::vector<std::vector<std::vector<double>>> LU = getLU(A);
    std::vector<std::vector<double>> L = LU[0];
    std::vector<std::vector<double>> U = LU[1];

    std::vector<double> z(n);
    z[0] = b[0];
    for (int i = 1; i < n; ++i) {
        double sum = 0;
        for (int j = 0; j < i; ++j) {
            sum += L[i][j] * z[j];
        }
        z[i] = b[i] - sum;
    }

    std::vector<double> x(n);
    x[n - 1] = z[n - 1] / U[n - 1][n - 1];
    for (int i = n - 2; i >= 0; --i) {
        double sum = 0;
        for (int j = i + 1; j < n; ++j) {
            sum += U[i][j] * x[j];
        }

        x[i] = (z[i] - sum) / U[i][i];
    }
    return x;
}

std::vector<double> MNK(std::vector<double> x, std::vector<double> y, int m) {
    std::vector<std::vector<double>> A(m + 1, std::vector<double>(m + 1));
    std::vector<double> b(m + 1);

    for (int k = 0; k < m + 1; ++k) {
        for (int j = 0; j < m + 1; ++j) {
            double sum = 0;
            for (int i = 0; i < x.size(); ++i) {
                sum += pow(x[i], k + j);
            }
            A[k][j] = sum;
        }

        double sum = 0;
        for (int i = 0; i < x.size(); ++i) {
            sum += y[i] * pow(x[i], k);
        }
        b[k] = sum;
    }

    std::vector<double> a = solution(A, b);
    return a;
}



int main() {
    std::cout << "Least squares method, 1st degree\n";
    std::vector<double> a = MNK(x, y, 1);
    std::cout << "Coefficients: [";
    for (int i = 0; i < a.size(); ++i) std::cout << " " << a[i] << " ";
    std::cout << "]\n";
    std::cout << "Sum of the error square: " << error(a, x, y) << "\n";

    std::cout << "2st degree\n";
    a = MNK(x, y, 2);
    std::cout << "Coefficients: [";
    for (int i = 0; i < a.size(); ++i) std::cout << " " << a[i] << " ";
    std::cout << "]\n";
    std::cout << "Sum of the error square: " << error(a, x, y) << "\n";

    return 0;
}