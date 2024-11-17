#include <iostream>
#include <vector>

void print_result(std::vector<double> res, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << res[i] << " ";
    }
}

void tridiagonalSolve(const std::vector<std::vector<double>>& A, const std::vector<double>& d, std::vector<double>& x) {
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

    for(int i = n - 1; i >= 0; --i) {
        if(i == n - 1) x[i] = Q[i];
        else {
            x[i] = P[i] * x[i + 1] + Q[i];
        }
    }
}

int main() {
    std::vector<std::vector<double>> A = {
            {-1, -1,  0, 0, 0},
            {7, -17, -8, 0, 0},
            {0, -9,  19, 8, 0},
            {0, 0, 7, -20, 4},
            {0, 0, 0, -4, 12}
    };
    std::vector<double> d = {-4, 132, -59, -193, -40};
    int n = A.size();

    std::vector<double> x(n);
    tridiagonalSolve(A, d, x);

    print_result(x, n);

    return 0;
}
