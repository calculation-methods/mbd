#include <iostream>
#include <cmath>
#include <vector>

const double EPSILON = 0.001;

double phi1(double x1, double x2) {
    return 1 + cos(x2);
}

double phi2(double x1, double x2) {
    return 1 + sin(x1);
}

double func1(double x1, double x2) {
    return x1 - cos(x2) - 1;
}

double func2(double x1, double x2) {
    return x2 - sin(x1) - 1;
}

double dfunc1x1(std::vector<double> x) {
    return 1;
}

double dfunc2x1(std::vector<double> x) {
    return -cos(x[0]);
}

double dfunc1x2(std::vector<double> x) {
    return sin(x[1]);
}

double dfunc2x2(std::vector<double> x) {
    return 1;
}

double norm(double x1, double x2) {
    return sqrt(x1 * x1 + x2 * x2);
}

std::vector<double> minus(std::vector<double> a, std::vector<double> b) {
    std::vector<double> d(a.size());
    for (int i = 0; i < a.size(); ++i) {
        d[i] = a[i] - b[i];
    }
    return d;
}

std::vector<double> solve(std::vector<std::vector<double>> A, std::vector<double> b) {
    int n = A.size();

    std::vector<std::vector<double>> B(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) B[i][j] = 0.0;
            else B[i][j] = -A[i][j] / A[i][i];
        }
    }

    std::vector<double> d(n);
    for (int i = 0; i < n; ++i) {
        d[i] = b[i] / A[i][i];
    }
    double eps = 0.0001;

    std::vector<double> xold(n, 1.0);
    std::vector<double> xnew = xold;
    double norm_val;
    do {
        xold = xnew;
        xnew = std::vector<double>(n);
        for (int i = 0; i < n; ++i) {
            double sum1 = 0;
            for (int j = 0; j < i; ++j) {
                sum1 += B[i][j] * xnew[j];
            }

            double sum2 = 0;
            for (int j = i; j < n; ++j) {
                sum2 += B[i][j] * xold[j];
            }
            xnew[i] = d[i] + sum1 + sum2;
        }
        norm_val = norm(minus(xnew, xold)[0], minus(xnew, xold)[1]);
    } while (norm_val > eps);
    return xnew;
}

int main() {
    double x1 = 0;
    double x2 = 3;

    double newX1 = phi1(x1, x2);
    double newX2 = phi2(x1, x2);

    while (norm(newX1 - x1, newX2 - x2) > EPSILON) {
        x1 = newX1;
        x2 = newX2;
        newX1 = phi1(x1, x2);
        newX2 = phi2(x1, x2);
    }

    std::cout << "Roots by simple iteration method: " << newX1 << " " << newX2 << std::endl;
    std::cout << "Check f1=" << func1(newX1, newX2) << " f2=" << func2(newX1, newX2) << std::endl;

    std::vector<double> x = { 0, 3 };

    std::vector<std::vector<double>> A(x.size(), std::vector<double>(x.size()));
    A[0][0] = dfunc1x1(x);
    A[0][1] = dfunc1x2(x);
    A[1][0] = dfunc2x1(x);
    A[1][1] = dfunc2x2(x);

    std::vector<double> dx = solve(A, std::vector<double>{-func1(x[0], x[1]), -func2(x[0], x[1])});
    std::vector<double> newX = { x[0] + dx[0], x[1] + dx[1] };

    while (norm(newX[0] - x[0], newX[1] - x[1]) > EPSILON) {
        x = newX;
        A[0][0] = dfunc1x1(x);
        A[0][1] = dfunc1x2(x);
        A[1][0] = dfunc2x1(x);
        A[1][1] = dfunc2x2(x);
        dx = solve(A, std::vector<double>{-func1(x[0], x[1]), -func2(x[0], x[1])});
        newX = { x[0] + dx[0], x[1] + dx[1] };
    }

    std::cout << "Roots by simple iteration method: " << newX[0] << " " << newX[1] << std::endl;
    std::cout << "Check f1=" << func1(newX[0], newX[1]) << " f2=" << func2(newX[0], newX[1]) << std::endl;
    return 0;
}
