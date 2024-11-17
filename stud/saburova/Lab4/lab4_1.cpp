#include "./lab4_1.h"

vector<double> f(double x, double y, double z) {
    vector<double> F(2);
    F[0] = z;
    F[1] = (exp(x*x) + 4*x*z - y)/(4*x*x-3);
    return F;
}

double calc_exact_y(double x) {
    return (exp(x) + exp(-x) - 1) * exp(x*x);
}

pair<vector<double>, vector<double>> method_adams(double x_start, double x_finish, double h, double y0, double z0) {
    int n = (x_finish - x_start) / h;
    vector<double> X(n + 1), Y(n + 1), Z(n + 1);
    for (int i = 0; i <= n; ++i) {
        X[i] = x_start + i * h;
    }
    vector<vector<double>> res_runge_kutta = method_runge_kutta_4(f, x_start, x_finish, h, y0, z0, 4);
    Y[0] = res_runge_kutta[1][0];
    Y[1] = res_runge_kutta[1][1];
    Y[2] = res_runge_kutta[1][2];
    Y[3] = res_runge_kutta[1][3];
    Z[0] = res_runge_kutta[2][0];
    Z[1] = res_runge_kutta[2][1];
    Z[2] = res_runge_kutta[2][2];
    Z[3] = res_runge_kutta[2][3];
    for (int i = 4; i <= n; ++i) {
        vector<double> F_k = f(X[i - 1], Y[i - 1], Z[i - 1]);
        vector<double> F_k_1 = f(X[i - 2], Y[i - 2], Z[i - 2]);
        vector<double> F_k_2 = f(X[i - 3], Y[i - 3], Z[i - 3]);
        vector<double> F_k_3 = f(X[i - 4], Y[i - 4], Z[i - 4]);
        Y[i] = Y[i - 1] + h / 24 * (55 * F_k[0] - 59 * F_k_1[0] + 37 * F_k_2[0] - 9 * F_k_3[0]);
        Z[i] = Z[i - 1] + h / 24 * (55 * F_k[1] - 59 * F_k_1[1] + 37 * F_k_2[1] - 9 * F_k_3[1]);
    }
    return {X, Y};
}

int main() {
    double start_pos = 0, end_pos = 1, h = 0.1;
    vector<vector<double>> res_h1 = method_runge_kutta_4(f, start_pos, end_pos, h, 1, 0);
    vector<double> X = res_h1[0];
    vector<double> Y_h1 = res_h1[1];
    vector<vector<double>> res_h2 = method_runge_kutta_4(f, start_pos, end_pos, h/2, 1, 0);
    vector<double> Y_h2 = res_h2[1];
    print(calc_exact_y, "Метод Рунге-Кутта:", X, Y_h1, Y_h2, 4);
    auto [X2_h1, Y2_h1] = method_adams(start_pos, end_pos, h, 1, 0);
    auto [X2_h2, Y2_h2] = method_adams(start_pos, end_pos, h/2, 1, 0);
    print(calc_exact_y, "\nМетод Адамса:", X2_h1, Y2_h1, Y2_h2, 4);
    return 0;
}