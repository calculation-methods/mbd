#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

double function(double x, double y, double z) {
    return -z / x - 2 * y / x;
}

double accurate_function(double x) {
    double sqrt_x = sqrt(x);
    return (cos(2) - sin(2)) * cos(2 * sqrt_x) + (cos(2) + sin(2)) * sin(2 * sqrt_x);
}

vector<double> Euler(double a, double b, double h) {
    int steps = (b - a) / h + 1;
    vector<double> x(steps);
    vector<double> y(steps, 1.0);
    vector<double> z(steps, 1.0);
    x[0] = a;
    for (int i = 1; i < steps; ++i) {
        x[i] = x[i - 1] + h;
        z[i] = z[i - 1] + h * function(x[i - 1], y[i - 1], z[i - 1]);
        y[i] = y[i - 1] + h * z[i - 1];
    }
    return y;
}

vector<vector <double>> Runge_Kutty(double a, double b, double h) {
    int steps = (b - a) / h + 1;
    vector<double> x(steps);
    vector<double> y(steps, 1);
    vector<double> z(steps, 1);
    vector<double> K(4, 0.0);
    vector<double> L(4, 0.0);
    x[0] = a;
    for (int i = 1; i < steps; ++i) {
        x[i] = x[i - 1] + h;
        for (int j = 1; j < K.size(); ++j) {
            K[0] = h * z[i - 1];
            L[0] = h * function(x[i - 1], y[i - 1], z[i - 1]);
            K[j] = h * (z[i - 1] + L[j - 1] / 2);
            L[j] = h * function(x[i - 1] + h / 2, y[i - 1] + K[j - 1] / 2, z[i - 1] + L[j - 1] / 2);
        }
        double deltay = (K[0] + 2 * K[1] + 2 * K[2] + K[3]) / 6;
        double deltaz = (L[0] + 2 * L[1] + 2 * L[2] + L[3]) / 6;
        y[i] = y[i - 1] + deltay;
        z[i] = z[i - 1] + deltaz;
    }
    return {y, z};
}


vector<double> Adams(double a, double b, double h) {
    int steps = (b - a) / h + 1;
    vector<double> x;
    vector<double> y(steps, 0);
    vector<double> z(steps, 0);
    for (double i = a; i < b+h; i += h) {
        x.push_back(i);
    }
    vector<double> y_start = Runge_Kutty(a, a + 3 * h, h)[0];
    vector<double> z_start = Runge_Kutty(a, a + 3 * h, h)[1];
    for (int i = 0; i < y_start.size(); ++i) {
        y[i] = y_start[i];
        z[i] = z_start[i];
    }
    for (int i = 4; i < steps; ++i) {
        z[i] = (z[i - 1] + h * (
                    55 * function(x[i - 1], y[i - 1], z[i - 1]) - 59 * function(x[i - 2], y[i - 2], z[i - 2])
                    + 37 * function(x[i - 3], y[i - 3], z[i - 3]) - 9 * function(x[i - 4], y[i - 4], z[i - 4])) / 24);
        y[i] = y[i - 1] + h * z[i - 1];
    }
    return y;
}

vector<vector<double>> RRR_method(double a, double b, double h) {
    vector<double> Euler1, Runge_Kutty1, Adams1;
    vector<double> Euler_norm = Euler(a, b, h);
    vector<double> Euler_half = Euler(a, b, h / 2);
    vector<double> Runge_Kutty_norm = Runge_Kutty(a, b, h)[0];
    vector<double> Runge_Kutty_half = Runge_Kutty(a, b, h / 2)[0];
    vector<double> Adams_norm = Adams(a, b, h);
    vector<double> Adams_half = Adams(a, b, h / 2);
    int steps = (b - a) / h + 1;
    Euler1.resize(steps);
    Runge_Kutty1.resize(steps);
    Adams1.resize(steps);
    for (int i = 0; i < steps; ++i) {
        Euler1[i] = Euler_norm[i] + (Euler_half[i * 2] - Euler_norm[i]) / (1 - 0.5 * 0.5);
        Runge_Kutty1[i] = Runge_Kutty_norm[i] + (Runge_Kutty_half[i * 2] - Runge_Kutty_norm[i]) / (1 - 0.5 * 0.5);
        Adams1[i] = Adams_norm[i] + (Adams_norm[i] - Adams_norm[i]) / (1 - 0.5 * 0.5);
    }
    return {Euler1, Runge_Kutty1, Adams1};
}


int main() {
    ofstream fout("answer1.txt");
    double h = 0.1;
    double a = 1;
    double b = 2;
    fout.precision(4);
    fout << fixed;
    vector<double> x, y;
    for (double i = a; i < b+h; i += h) {
        x.push_back(i);
        y.push_back(accurate_function(i));
    }
    fout << "         X:        ";
    for (int i = 0; i < x.size(); ++i) {
        fout << x[i] << " ";
    }
    fout << endl;
    fout << "accurate value Y: ";
    for (int i = 0; i < y.size(); ++i) {
        fout << y[i] << " ";
    }
    fout << endl << endl;
    fout << "    Euler method   ";
    for (int i = 0; i < Euler(a, b, h).size(); ++i) {
        fout << Euler(a, b, h)[i] << " ";
    }
    fout << endl;
    fout << " Runge-Kutte method ";
    for (int i = 0; i < Runge_Kutty(a, b, h)[0].size(); ++i) {
        fout << Runge_Kutty(a, b, h)[0][i] << " ";
    }
    fout << endl;
    fout << "   Adams method    ";
    vector <double> res_adams = Adams(a, b, h);
    for (int i = 0; i < res_adams.size(); ++i) {
        fout << res_adams[i] << " ";
    }
    fout << endl << endl;
    fout << "With Runge-Romberg-Richardson method: \n" << endl;
    vector<vector<double>> result = RRR_method(a, b, h);
    fout << "    Euler method   ";
    for (int i = 0; i < result[0].size(); ++i) {
        fout << result[0][i] << " ";
    }
    fout << endl;
    fout << " Runge-Kutte method ";
    for (int i = 0; i < result[1].size(); ++i) {
        fout << result[1][i] << " ";
    }
    fout << endl;
    fout << "   Adams method    ";
    for (int i = 0; i < result[2].size(); ++i) {
        fout << result[2][i] << " ";
    }
    fout << endl << endl;
    fout << "Delta of accurate value \n" << endl;
    fout << "    Euler method   ";
    for (int i = 0; i < result[0].size(); ++i) {
        fout << abs(result[0][i] - y[i]) << " ";
    }
    fout << endl;
    fout << " Runge-Kutte method ";
    for (int i = 0; i < result[1].size(); ++i) {
        fout << abs(result[1][i] - y[i]) << " ";
    }
    fout << endl;
    fout << "   Adams method    ";
    for (int i = 0; i < result[2].size(); ++i) {
        fout << abs(result[2][i] - y[i]) << " ";
    }
    fout << endl;
    return 0;
}