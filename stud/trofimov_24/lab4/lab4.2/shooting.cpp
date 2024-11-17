#include <vector>
#include <cmath>
#include "shooting.h"
#include "funcs.h"



using namespace std;

vector<vector<double>>  Shooting::RungeKutta(function<vector<double>(double, const vector<double>&)> f,
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

double  Shooting::ShootingFunc(double s, double x_end, int N) {
    vector<double> Y0 = {0, s};
    auto sol = RungeKutta(odes, 0, Y0, x_end, N);
    double y1 = sol.back()[0];
    return y1 -3 - M_PI/2;

}

vector<vector<double>> Shooting::result(double s_guess, double x_end, int N) {
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

    return RungeKutta(odes, 0, {0, s}, x_end, N);
}