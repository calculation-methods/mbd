#include <bits/stdc++.h>

using namespace std;

double method_runge_romberg(double y1, double y2, int64_t p) {
    return (y1 - y2) / (pow(2, p) - 1);
}

vector<double> num_vector(vector<double> a, double n) {
    for (int i = 0; i < a.size(); ++i) {
        a[i] *= n;
    }
    return a;
}

vector<vector<double>> method_runge_kutta_4(vector<double> (*f)(double, double, double), double x_start, double x_finish, double h, double y0, double z0, int iter = -1) {
    int n;
    if (iter == -1) {
        n = (x_finish - x_start) / h;
    } else
        n = iter;
    vector<double> X(n + 1), Y(n + 1), Z(n + 1);
    for (int i = 0; i <= n; ++i) {
        X[i] = x_start + i * h;
    }
    Y[0] = y0;
    Z[0] = z0;
    for (int i = 1; i <= n; ++i) {
        vector<double> K_1 = num_vector(f(X[i - 1], Y[i - 1], Z[i - 1]), h);
        vector<double> K_2 = num_vector(f(X[i - 1] + h / 2, Y[i - 1] + K_1[0] / 2, Z[i - 1] + K_1[1] / 2), h);
        vector<double> K_3 = num_vector(f(X[i - 1] + h / 2, Y[i - 1] + K_2[0] / 2, Z[i - 1] + K_2[1] / 2), h);
        vector<double> K_4 = num_vector(f(X[i - 1] + h, Y[i - 1] + K_3[0], Z[i - 1] + K_3[1]), h);
        Y[i] = Y[i - 1] + 1.0 / 6 * (K_1[0] + 2 * K_2[0] + 2 * K_3[0] + K_4[0]);
        Z[i] = Z[i - 1] + 1.0 / 6 * (K_1[1] + 2 * K_2[1] + 2 * K_3[1] + K_4[1]);
    }
    return {X, Y, Z};
}

void print(double (*calc_exact_y)(double), string method, vector<double> &X_h1, vector<double> &Y_h1, vector<double> &Y_h2, int64_t p) {
    cout << method << "\n"
         << "  x  |" << " y  |" << "точное значенние y |" << "\tепсилон\t   | Рунге-Ромберг\n-------------------------------------------------------\n";
    for (int i = 0; i < X_h1.size(); ++i) {
        double exact_y = calc_exact_y(X_h1[i]);
        cout << setprecision(9) << " " << X_h1[i] << " | " << Y_h1[i] << " | " << exact_y << " | " << abs(exact_y - Y_h1[i]) << " | " << method_runge_romberg(Y_h1[i], Y_h2[2 * i], p) << endl;
    }
}