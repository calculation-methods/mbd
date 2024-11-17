#include <vector>
#include <cmath>
#include "finite_diff.h"

using namespace std;


vector<double> FiniteDifference::result(int N) {
    double h = 1.0 / N;
    vector<double> x(N + 1);
    vector<double> A(N + 1);
    vector<double> B(N + 1);
    vector<double> C(N + 1);
    vector<double> D(N + 1);
    vector<double> y(N + 1, 0.0);

    for (int i = 0; i <= N; ++i) {
        x[i] = i * h;
    }

    for (int i = 1; i < N; ++i) {
        double xi = x[i];
        A[i] = (xi * xi - 1) / (h * h) - (xi - 3) / (2 * h);
        B[i] = -2 * (xi * xi - 1) / (h * h);
        C[i] = (xi * xi - 1) / (h * h) + (xi - 3) / (2 * h);
        D[i] = 0;
    }

    B[0] = 1; D[0] = 2;
    B[N] = 1 + h; D[N] = (3 + M_PI/2) * h;

    for (int i = 1; i <= N; ++i) {
        double m = A[i] / B[i - 1];
        B[i] -= m * C[i - 1];
        D[i] -= m * D[i - 1];
    }

    y[N] = D[N] / B[N];
    for (int i = N - 1; i >= 0; --i) {
        y[i] = (D[i] - C[i] * y[i + 1]) / B[i];
    }

    return y;
}