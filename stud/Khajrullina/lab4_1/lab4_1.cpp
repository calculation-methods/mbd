#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

double f(double x, double y, double z) {
    return -2 * z / x - y / pow(x, 4);
}

double exact_solution(double x) {
    return (sin(1) + cos(1)) * cos(1 / x) + (sin(1) - cos(1)) * sin(1 / x);
}

vector<double> Euler_method(double a, double b, double h) {
    int s = (b - a) / h + 1;
    vector<double> x(s);
    vector<double> y(s, 1.0);
    vector<double> z(s, 1.0);
    x[0] = a;
    for (int i = 1; i < s; ++i) {
        x[i] = x[i - 1] + h;
        z[i] = z[i - 1] + h * f(x[i - 1], y[i - 1], z[i - 1]);
        y[i] = y[i - 1] + h * z[i - 1];
    }
    return y;
}

vector<vector <double>> RK_method(double a, double b, double h) {
    int s = (b - a) / h + 1;
    vector<double> x(s);
    vector<double> y(s, 1);
    vector<double> z(s, 1);
    vector<double> K(4, 0.0);
    vector<double> L(4, 0.0);
    x[0] = a;
    for (int i = 1; i < s; ++i) {
        x[i] = x[i - 1] + h;
        for (int j = 1; j < K.size(); ++j) {
            K[0] = h * z[i - 1];
            L[0] = h * f(x[i - 1], y[i - 1], z[i - 1]);
            K[j] = h * (z[i - 1] + L[j - 1] / 2);
            L[j] = h * f(x[i - 1] + h / 2, y[i - 1] + K[j - 1] / 2, z[i - 1] + L[j - 1] / 2);
        }
        double deltay = (K[0] + 2 * K[1] + 2 * K[2] + K[3]) / 6;
        double deltaz = (L[0] + 2 * L[1] + 2 * L[2] + L[3]) / 6;
        y[i] = y[i - 1] + deltay;
        z[i] = z[i - 1] + deltaz;
    }
    return { y, z };
}


vector<double> Adams_method(double a, double b, double h) {
    int s = (b - a) / h + 1;
    vector<double> x;
    vector<double> y(s, 0);
    vector<double> z(s, 0);
    for (double i = a; i < b + h; i += h) {
        x.push_back(i);
    }
    vector<double> y_start = RK_method(a, a + 3 * h, h)[0];
    vector<double> z_start = RK_method(a, a + 3 * h, h)[1];
    for (int i = 0; i < y_start.size(); ++i) {
        y[i] = y_start[i];
        z[i] = z_start[i];
    }
    for (int i = 4; i < s; ++i) {
        z[i] = (z[i - 1] + h * (
            55 * f(x[i - 1], y[i - 1], z[i - 1]) - 59 * f(x[i - 2], y[i - 2], z[i - 2])
            + 37 * f(x[i - 3], y[i - 3], z[i - 3]) - 9 * f(x[i - 4], y[i - 4], z[i - 4])) / 24);
        y[i] = y[i - 1] + h * z[i - 1];
    }
    return y;
}

vector<vector<double>> RRR_method(double a, double b, double h) {
    vector<double> Euler1, RK1, Adams1;
    vector<double> Euler_norm = Euler_method(a, b, h);
    vector<double> Euler_half = Euler_method(a, b, h / 2);
    vector<double> RK_norm = RK_method(a, b, h)[0];
    vector<double> RK_half = RK_method(a, b, h / 2)[0];
    vector<double> Adams_norm = Adams_method(a, b, h);
    vector<double> Adams_half = Adams_method(a, b, h / 2);
    int s = (b - a) / h + 1;
    Euler1.resize(s);
    RK1.resize(s);
    Adams1.resize(s);
    for (int i = 0; i < s; ++i) {
        Euler1[i] = Euler_norm[i] + (Euler_half[i * 2] - Euler_norm[i]) / (1 - 0.5 * 0.5);
        RK1[i] = RK_norm[i] + (RK_half[i * 2] - RK_norm[i]) / (1 - 0.5 * 0.5);
        Adams1[i] = Adams_norm[i] + (Adams_norm[i] - Adams_norm[i]) / (1 - 0.5 * 0.5);
    }
    return { Euler1, RK1, Adams1 };
}


int main() {
    double h = 0.1;
    double a = 1;
    double b = 2;
    vector<double> x, y;
    for (double i = a; i < b + h; i += h) {
        x.push_back(i);
        y.push_back(exact_solution(i));
    }
    std::cout << "X: ";
    for (int i = 0; i < x.size(); ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Y: ";
    for (int i = 0; i < y.size(); ++i) {
        std::cout << y[i] << " ";
    }
    std::cout << endl << std::endl;
    std::cout << "Euler method: ";
    for (int i = 0; i < Euler_method(a, b, h).size(); ++i) {
        std::cout << Euler_method(a, b, h)[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Runge-Kutte method: ";
    for (int i = 0; i < RK_method(a, b, h)[0].size(); ++i) {
        std::cout << RK_method(a, b, h)[0][i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Adams method: ";
    vector <double> res_adams = Adams_method(a, b, h);
    for (int i = 0; i < res_adams.size(); ++i) {
        std::cout << res_adams[i] << " ";
    }
    std::cout << endl << std::endl;
    std::cout << "With Runge-Romberg-Richardson method \n" << std::endl;
    vector<vector<double>> result = RRR_method(a, b, h);
    std::cout << "Euler method: ";
    for (int i = 0; i < result[0].size(); ++i) {
        std::cout << result[0][i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Runge-Kutte method: ";
    for (int i = 0; i < result[1].size(); ++i) {
        std::cout << result[1][i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Adams method: ";
    for (int i = 0; i < result[2].size(); ++i) {
        std::cout << result[2][i] << " ";
    }
    std::cout << endl << std::endl;
    std::cout << "Delta of exact value \n" << std::endl;
    std::cout << "Euler method: ";
    for (int i = 0; i < result[0].size(); ++i) {
        std::cout << abs(result[0][i] - y[i]) << " ";
    }
    std::cout << std::endl;
    std::cout << "Runge-Kutte method: ";
    for (int i = 0; i < result[1].size(); ++i) {
        std::cout << abs(result[1][i] - y[i]) << " ";
    }
    std::cout << std::endl;
    std::cout << "Adams method: ";
    for (int i = 0; i < result[2].size(); ++i) {
        std::cout << abs(result[2][i] - y[i]) << " ";
    }
    std::cout << std::endl;
    return 0;
}