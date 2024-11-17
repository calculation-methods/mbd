#include <iostream>
#include <vector>
#include <iomanip>

void printMatrix(const std::vector<std::vector<double>>& matrix) {
    for (const auto& m : matrix) {
        std::cout << "|";
        for (const auto& d : m) std::cout << "\t" << std::fixed << std::setprecision(2) << d << "\t";
        std::cout << "|\n";
    }
}

int main() {
    std::vector<std::vector<double>> A = {
        {-7.0, -6.0, 0.0, 0.0, 0.0},
        {6.0, 12.0, 0.0, 0.0, 0.0},
        {0.0, -3.0, 5.0, 0.0, 0.0},
        {0.0, 0.0, -9.0, 21.0, 8.0},
        {0.0, 0.0, 0.0, -5.0, -6.0}
    };
    std::vector<double> d = {-75.0, 126.0, 13.0, -40.0, -24.0};
    int n = A.size();

    std::vector<double> P(n);
    std::vector<double> Q(n);

    for(int i = 0; i < n; ++i) {
        if(i == 0) {
            P[i] = -A[i][i+1] / A[i][i];
            Q[i] = d[i] / A[i][i];
        } else if(i == n - 1) {
            P[i] = 0;
            Q[i] = (d[i] - A[i][i - 1] * Q[i - 1]) / (A[i][i] + A[i][i - 1] * P[i - 1]);
        } else {
            P[i] = -A[i][i+1] / (A[i][i] + A[i][i - 1] * P[i - 1]);
            Q[i] = (d[i] - A[i][i - 1] * Q[i - 1]) / (A[i][i] + A[i][i - 1] * P[i - 1]);
        }
    }

    std::vector<double> x(n);

    for(int i = n - 1; i >= 0; --i) {
        if(i == n - 1) x[i] = Q[i];
        else {
            x[i] = P[i] * x[i + 1] + Q[i];
        }
    }

    std::cout << "Result x\n";
    printMatrix({x});

    return 0;
}