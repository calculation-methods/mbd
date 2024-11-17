#include <cmath>
#include <vector>
#include <iostream>
#include <utility>
#include <functional>

using namespace std;
using vect = vector<double>;


vect f(vect& x) {
    vect res;
    for (double val : x)
        res.push_back((1 + exp(val * val / 2)) / val);
    return res;
}


double ddf(double x, double f, double df) {
    return (x * (x * x - 1) * df + (x * x + 1) * f) / (x * x);
}


double sqr_error(const vect& a, const vect& b) {
    double res = 0;
    for (int i = 0; i < a.size(); i++)
        res += pow(a[i] - b[i], 2);
    return sqrt(res);
}

vect Euler(function<double(double, double, double)> ddy, pair<double, double>& borders, double y0, double z0, double h) {
    double x = borders.first;
    int N = (borders.second - borders.first) / h + 1;
    vect y(N), z(N);

    y[0] = y0;
    z[0] = z0;

    for (int i = 0; i < N - 1; i++) {
        y[i + 1] = y[i] + h * z[i];
        z[i + 1] = z[i] + h * ddy(x, y[i], z[i]);
        x += h;
    }

    return y;
}


pair<vect, vect> RungeKutta(function<double(double, double, double)> ddy, pair<double, double>& borders, double y0, double z0, double h) {
    double x = borders.first;
    int N = (borders.second - borders.first) / h + 1;
    vect y(N), z(N);

    y[0] = y0;
    z[0] = z0;

    for (int i = 0; i < N - 1; i++) {
        double K1 = h * z[i];
        double L1 = h * ddy(x, y[i], z[i]);
        double K2 = h * (z[i] + 0.5 * L1);
        double L2 = h * ddy(x + 0.5 * h, y[i] + 0.5 * K1, z[i] + 0.5 * L1);
        double K3 = h * (z[i] + 0.5 * L2);
        double L3 = h * ddy(x + 0.5 * h, y[i] + 0.5 * K2, z[i] + 0.5 * L2);
        double K4 = h * (z[i] + L3);
        double L4 = h * ddy(x + h, y[i] + K3, z[i] + L3);
        double delta_y = (K1 + 2 * K2 + 2 * K3 + K4) / 6;
        double delta_z = (L1 + 2 * L2 + 2 * L3 + L4) / 6;
        y[i + 1] = y[i] + delta_y;
        z[i + 1] = z[i] + delta_z;
        x += h;
    }

    return { y, z };
}


vect Adams(function<double(double, double, double)> ddy, pair<double, double>& borders, vect calc_y, vect& calc_z, double h) {
    double x = borders.first;
    int N = (borders.second - borders.first) / h + 1;
    vect y, z;

    for (int i = 0; i < 4; i++) {
        y.push_back(calc_y[i]);
        z.push_back(calc_z[i]);
    }

    for (int i = 3; i < N - 1; i++) {
        double z_next = z[i] + (h / 24) * (55 * ddy(x, y[i], z[i]) - 59 * ddy(x - h, y[i - 1], z[i - 1]) + 37 * ddy(x - 2 * h, y[i - 2], z[i - 2]) - 9 * ddy(x - 3 * h, y[i - 3], z[i - 3]));
        double y_next = y[i] + (h / 24) * (55 * z[i] - 59 * z[i - 1] + 37 * z[i - 2] - 9 * z[i - 3]);
        double z_i = z[i] + (h / 24) * (9 * ddy(x + h, y_next, z_next) + 19 * ddy(x, y[i], z[i]) - 5 * ddy(x - h, y[i - 1], z[i - 1]) + 1 * ddy(x - 2 * h, y[i - 2], z[i - 2]));
        z.push_back(z_i);
        double y_i = y[i] + (h / 24) * (9 * z_next + 19 * z[i] - 5 * z[i - 1] + 1 * z[i - 2]);
        y.push_back(y_i);
        x += h;
    }

    return y;
}


vect RungeRomberg(vect y1, vect y2, double h1, double h2, double p) {
    if (h1 > h2) {
        int k = h1 / h2;
        vect y(y1.size());
        for (int i = 0; i < y1.size(); i++)
            y[i] = y2[i * k] + (y2[i * k] - y1[i]) / (pow(k, p) - 1);
        return y;
    }
    else {
        int k = h2 / h1;
        vect y(y2.size());
        for (int i = 0; i < y2.size(); i++)
            y[i] = y1[i * k] + (y1[i * k] - y2[i]) / (pow(k, p) - 1);
        return y;
    }
}


int main() {
    pair<double, double> borders = { 1, 2 };
    double y0 = 2.649, z0 = -1.0, h = 0.1;
    double h2 = h / 2;
    vect x = { borders.first };
    double last_el = borders.first;
    while (last_el < borders.second) {
        last_el += h;
        x.push_back(last_el);
    }

    vect y = f(x);

    vect y1, y2, y3, z, z2, y1_2, y2_2, y3_2;
    y1 = Euler(ddf, borders, y0, z0, h);
    tie(y2, z) = RungeKutta(ddf, borders, y0, z0, h);
    y3 = Adams(ddf, borders, y2, z, h);
    y1_2 = Euler(ddf, borders, y0, z0, h2);
    tie(y2_2, z2) = RungeKutta(ddf, borders, y0, z0, h2);
    y3_2 = Adams(ddf, borders, y2_2, z2, h2);
    cout << "Error estimation by the Runge-Rombert method:" << endl;
    cout << "For the explicit Euler method: " << sqr_error(y1, RungeRomberg(y1, y1_2, h, h2, 1)) << endl;
    cout << "For the Runge-Kutta method: " << sqr_error(y2, RungeRomberg(y2, y2_2, h, h2, 4)) << endl;
    cout << "For the Adams method: " << sqr_error(y3, RungeRomberg(y3, y3_2, h, h2, 4)) << endl;
    cout << "Comparison with the exact solution:" << endl;
    cout << "For the explicit Euler method: " << sqr_error(y1, y) << endl;
    cout << "For the Runge-Kutta method: " << sqr_error(y2, y) << endl;
    cout << "For the Adams method: " << sqr_error(y3, y) << endl;
}