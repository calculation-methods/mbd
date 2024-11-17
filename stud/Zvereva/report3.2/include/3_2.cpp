#include <iostream>
#include <vector>
#include <cmath>

std::vector<double> x = { -3, -1, 1, 3, 5 };
std::vector<double> y = { -1.2490,    -0.78540,    0.78540,    1.2490,    1.3734 };

std::vector<std::vector<double>> createA(std::vector<double> x) {
    int n = x.size();
    int size = n - 2;
    std::vector<std::vector<double>> A(size, std::vector<double>(size, 0));

    A[0][0] = 2 * ((x[1] - x[0]) + (x[2] - x[1]));
    A[0][1] = x[2] - x[1];
    A[size - 1][size - 2] = x[n - 2] - x[n - 3];
    A[size - 1][size - 1] = 2 * ((x[n - 2] - x[n - 3]) + (x[n - 1] - x[n - 2]));

    for (int i = 3; i < n - 1; ++i) {
        A[i - 2][i - 3] = x[i - 1] - x[i - 2];
        A[i - 2][i - 2] = 2 * (x[i - 1] - x[i - 2] + x[i] - x[i - 1]);
        A[i - 2][i - 1] = (x[i] - x[i - 1]);
    }

    return A;
}

std::vector<double> createB(std::vector<double> x, std::vector<double> y) {
    int n = x.size();
    int size = n - 2;
    std::vector<double> b(size);

    b[0] = 3 * ((y[2] - y[1]) / (x[2] - x[1]) - (y[1] - y[0]) / (x[1] - x[0]));
    b[size - 1] = 3 * ((y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]) - (y[n - 2] - y[n - 3]) / (x[n - 2] - x[n - 3]));

    for (int i = 3; i < n - 1; ++i) {
        b[i - 2] = 3 * ((y[i] - y[i - 1]) / (x[i] - x[i - 1]) - (y[i - 1] - y[i - 2]) / (x[i - 1] - x[i - 2]));
    }
    return b;
}

std::vector<double> solve(std::vector<std::vector<double>> A, std::vector<double> d) {
    int n = A.size();

    std::vector<double> P(n);
    std::vector<double> Q(n);

    for (int i = 0; i < n; ++i) {
        if (i == 0) {
            P[i] = -A[i][i + 1] / A[i][i];
            Q[i] = d[i] / A[i][i];
        }
        else if (i == n - 1) {
            P[i] = 0;
            Q[i] = (d[i] - A[i][i - 1] * Q[i - 1]) / (A[i][i] + A[i][i - 1] * P[i - 1]);
        }
        else {
            P[i] = -A[i][i + 1] / (A[i][i] + A[i][i - 1] * P[i - 1]);
            Q[i] = (d[i] - A[i][i - 1] * Q[i - 1]) / (A[i][i] + A[i][i - 1] * P[i - 1]);
        }
    }

    std::vector<double> x(n);

    for (int i = n - 1; i >= 0; --i) {
        if (i == n - 1) x[i] = Q[i];
        else {
            x[i] = P[i] * x[i + 1] + Q[i];
        }
    }
    return x;
}

int main() {
    double X = -0.5;
    int n = x.size();

    std::vector<std::vector<double>> A = createA(x);
    std::vector<double> B = createB(x, y);

    std::vector<double> solveC = solve(A, B);
    std::vector<double> c(n - 1, 0);
    for (int i = 1; i < c.size(); ++i) c[i] = solveC[i - 1];

    std::vector<double> a(n - 1);
    for (int i = 1; i < n; ++i) a[i - 1] = y[i - 1];

    std::vector<double> b(n - 1);
    for (int i = 1; i < n - 1; ++i) {
        b[i - 1] = (y[i] - y[i - 1]) / (x[i] - x[i - 1]) - (x[i] - x[i - 1]) / 3 * (c[i] + 2 * c[i - 1]);
    }
    b[n - 2] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]) - (x[n - 1] - x[n - 2]) / 3 * 2 * c[n - 2];

    std::vector<double> d(n - 1);
    for (int i = 1; i < n - 1; ++i) {
        d[i - 1] = (c[i] - c[i - 1]) / (3 * (x[i] - x[i - 1]));
    }
    d[n - 2] = -c[n - 2] / 3 / (x[n - 1] - x[n - 2]);

    //finding Y
    double Y = 0.0;
    for (int i = 0; i < n - 1; ++i)
        if (x[i] < X & x[i + 1] > X)
            Y = a[i] + b[i] * (X - x[i]) + c[i] * pow((X - x[i]), 2) + d[i] * pow((X - x[i]), 3);

    std::cout << "Y in  X* by spline is: " << Y << "\n";

    return 0;
}
