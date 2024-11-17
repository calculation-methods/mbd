#include "./matrix.h"
#include "./lab4_1.h"

vector<double> f(double x, double y, double z) {
    vector<double> F(2);
    F[0] = z;
    F[1] = ((2*x+1)*y + 4*x*z)/4;
    return F;
}

double calc_exact_y(double x) {
    return 3*x + exp(-2*x*x);
}

double phi(double x_start, double x_finish, double h, double y0, double n, double y1) {
    vector<vector<double>> res = method_runge_kutta_4(f, x_start, x_finish, h, y0, n);
    return res[1].back() - y1;
}

vector<vector<double>> shooting(double x_start, double x_finish, double h, double y0, double y1, double n0, double n1, double eps) {
    double ni = n0, ni_1 = n1;
    double phi_ni = phi(x_start, x_finish, h, y0, ni, y1);
    double phi_ni_1 = phi(x_start, x_finish, h, y0, ni_1, y1);
    while (abs(phi_ni_1) > eps) {
        double ni_2 = ni_1 - (ni_1 - ni) / (phi_ni_1 - phi_ni) * phi_ni_1;
        ni = ni_1;
        ni_1 = ni_2;
        phi_ni = phi_ni_1;
        phi_ni_1 = phi(x_start, x_finish, h, y0, ni_1, y1);
    }
    return method_runge_kutta_4(f, x_start, x_finish, h, y0, ni_1);
}

double p(double x) {
    return (x - 3) / (x * x - 1);
}

double q(double x) {
    return -1 / (x * x - 1);
}

double f2(double x) {
    return 0;
}

vector<vector<double>> finite_difference(double x_start, double x_finish, double y0, double y1, double h) {
    int n = (x_finish - x_start) / h + 1;
    vector<double> X(n);
    for (int i = 0; i < n; ++i) {
        X[i] = x_start + i * h;
    }
    Matrix A(n, n), B(n, 1);
    A(0, 0) = h;
    A(0, 1) = 0;
    for (int i = 1; i < n - 1; ++i) {
        A(i, i - 1) = 1 - p(X[i]) * h / 2;
        A(i, i) = -2 + h * h * q(X[i]);
        A(i, i + 1) = 1 + p(X[i]) * h / 2;
    }
    A(n - 1, n - 2) = 0;
    A(n - 1, n - 1) = h;
    B(0, 0) = h * y0;
    for (int i = 1; i < n - 1; ++i) {
        B(i, 0) = h * h * f2(X[i]);
    }
    B(n - 1, 0) = h * y1;
    Matrix C = A.run_through_method(B);
    vector<double> Y;
    for (int i = 0; i < C.GetRows(); ++i) {
        Y.push_back(C(i, 0));
    }
    return {X, Y};
}

int main() {
    double h1 = 0.1, h2 = 0.05;
    vector<vector<double>> res = shooting(2, 3, h1, -6, 0, 5, 7, 0.000001);
    vector<double> X = res[0];
    vector<double> Y_h1 = res[1];
    res = shooting(2, 3, h2, -6, 0, 5, 7, 0.000001);
    vector<double> Y_h2 = res[1];
    print(calc_exact_y, "Method shooting:", X, Y_h1, Y_h2, 4);
    vector<vector<double>> res2 = finite_difference(2, 3, -6, 0, h1);
    X = res2[0];
    vector<double> Y2_h1 = res2[1];
    res2 = finite_difference(2, 3, -6, 0, h2);
    Y_h2 = res2[1];
    print(calc_exact_y, "\nFinite difference method:", X, Y2_h1, Y_h2, 2);
    vector<double> exact_y;
    for (int i = 0; i < X.size(); ++i) {
        exact_y.push_back(calc_exact_y(X[i]));
    }
    return 0;
}