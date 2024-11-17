#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <iomanip>
#include <fstream>

using namespace std;


double ExactSolution(double x) {
    return x - 3 + 1 / (x + 1);
}

vector<double> Odes(double x, const vector<double>& Y) {
    double y = Y[0];
    double dy = Y[1];
    double d2y = (y - (x - 3) * dy) / (x * x - 1);
    return {dy, d2y};
}

vector<vector<double>> RungeKutta(function<vector<double>(double, const vector<double>&)> f,
                                     double x0, const vector<double>& Y0, double xf, int N) {
    vector<vector<double>> result;
    result.push_back(Y0);
    double h = (xf - x0) / N;
    double x = x0;
    vector<double> Y = Y0;
    for (int i = 1; i <= N; ++i) {
        vector<double> k1 = f(x, Y);
        vector<double> k2 = f(x + h / 2, {Y[0] + h / 2 * k1[0], Y[1] + h / 2 * k1[1]});
        vector<double> k3 = f(x + h / 2, {Y[0] + h / 2 * k2[0], Y[1] + h / 2 * k2[1]});
        vector<double> k4 = f(x + h, {Y[0] + h * k3[0], Y[1] + h * k3[1]});
        Y[0] += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
        Y[1] += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
        x += h;
        result.push_back(Y);
    }
    return result;
}

double ShootingFunc(double s, double x_end, int N) {
    vector<double> Y0 = {0, s};
    auto sol = RungeKutta(Odes, 0, Y0, x_end, N);
    double y1 = sol.back()[0];
    double dy1 = sol.back()[1];
    return dy1 + y1 + 0.75;
}

vector<vector<double>> Shooting(double s_guess, double x_end, int N) {
    double s = s_guess;
    double epsilon = 1e-6;
    double delta = 1e-4;
    double f_s = ShootingFunc(s, x_end, N);

    while (abs(f_s) > epsilon) {
        double f_s_delta = ShootingFunc(s + delta, x_end, N);
        double s_new = s - f_s * delta / (f_s_delta - f_s);
        s = s_new;
        f_s = ShootingFunc(s, x_end, N);
    }

    return RungeKutta(Odes, 0, {0, s}, x_end, N);
}

vector<double> FiniteDifference(int N) {
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

    B[0] = 1; D[0] = 0; // y'(0) = 0
    B[N] = 1 + h; A[N] = -1; D[N] = -0.75 * h; // y'(1) + y(1) = -0.75

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

double RungeRomberg(const vector<double>& y2h, const vector<double>& yh, int N) {
    double error = 0.0;
    for (int i = 0; i <= N; ++i) {
        error = max(error, abs(y2h[2 * i] - yh[i]) / 3.0);
    }
    return error;
}

int main() {
    ofstream fout("output.txt");
    fout << fixed << setprecision(6);
    int N = 100;
    double x_end = 1.0;

    double s_guess = -1;
    auto sol_shooting = Shooting(s_guess, x_end, N);
    auto sol_shooting_2N = Shooting(s_guess, x_end, 2 * N);

    auto sol_fd = FiniteDifference(N);
    auto sol_fd_2N = FiniteDifference(2 * N);

    vector<double> x(N + 1);
    vector<double> exact(N + 1);
    for (int i = 0; i <= N; ++i) {
        x[i] = i * 1.0 / N;
        exact[i] = ExactSolution(x[i]);
    }

    double error_fd = RungeRomberg(sol_fd_2N, sol_fd, N);
    vector<double> y_shooting(N + 1);
    vector<double> y_shooting_2N(2 * N + 1);
    for (int i = 0; i <= N; ++i) {
        y_shooting[i] = sol_shooting[i][0];
    }
    for (int i = 0; i <= 2 * N; ++i) {
        y_shooting_2N[i] = sol_shooting_2N[i][0];
    }
    double error_shooting = RungeRomberg(y_shooting_2N, y_shooting, N);

    fout << "Метод стрельбы:\n";
    fout << "x = " << x[N] << ", y = " << sol_shooting[N][0] << ", точное y = " << exact[N] << "\n";

    fout << "Погрешность метода стрельбы (метод Рунге-Ромберга): " << error_shooting << "\n";

    fout << "\nМетод конечных разностей:\n";
    fout << "x = " << x[N] << ", y = " << sol_fd[N] << ", точное y = " << exact[N];

    fout << "\nПогрешность метода конечных разностей (метод Рунге-Ромберга): " << error_fd << "\n";

    return 0;
}
