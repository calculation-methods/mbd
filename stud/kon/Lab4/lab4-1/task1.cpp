#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

double exact_solution(double x){
    return x - x*x + 1;
}

void euler(double h, double x_0, double y1_0, double y2_0, double end, vector<double>& x_val, vector<double>& y1_val, vector<double>& y2_val) {
    double x = x_0;
    double y1 = y1_0;
    double y2 = y2_0;
    while (x <= end) {
        x_val.push_back(x);
        y1_val.push_back(y1);
        y2_val.push_back(y2);
        double y1_new = y1 + h * y2;
        double y2_new = y2 + h * ((2 * x * y2 - 2 * y1)/(x * x + 1));
        y1 = y1_new;
        y2 = y2_new;
        x += h;
    }
}

void runge_kutta(double h, double x_0, double y1_0, double y2_0, double end, vector<double>& x_val, vector<double>& y1_val, vector<double>& y2_val) {
    double x = x_0;
    double y1 = y1_0;
    double y2 = y2_0;
    while (x <= end) {
        x_val.push_back(x);
        y1_val.push_back(y1);
        y2_val.push_back(y2);
        double k1_1 = h * y2;
        double k1_2 = h * ((2 * x * y2 - 2 * y1)/(x * x + 1));

        double k2_1 = h * (y2 + 0.5 * k1_2);
        double k2_2 = h * ((2 * (x + 0.5 * h) * (y2 + 0.5 * k1_2) - 2 * (y1 + 0.5 * k1_1)) / ((x + 0.5 * h) * (x + 0.5 * h) + 1));

        double k3_1 = h * (y2 + 0.5 * k2_2);
        double k3_2 = h * ((2 * (x + 0.5 * h) * (y2 + 0.5 * k2_2) - 2 * (y1 + 0.5 * k2_1)) / ((x + 0.5 * h) * (x + 0.5 * h) + 1));

        double k4_1 = h * (y2 + k3_2);
        double k4_2 = h * ((2 * (x + h) * (y2 + k3_2) - 2 * (y1 + k3_1)) / ((x + h) * (x + h) + 1));

        y1 += (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1) / 6;
        y2 += (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2) / 6;
        x += h;
    }
}

void adam(double h, double x_0, double y1_0, double y2_0, double end, vector<double>& x_val, vector<double>& y1_val, vector<double>& y2_val) {
    double x = x_0;
    double y1 = y1_0;
    double y2 = y2_0;
    vector<double> f_1, f_2;
    for (int i = 0; i < 4; i++) {
        x_val.push_back(x);
        y1_val.push_back(y1);
        y2_val.push_back(y2);

        double k1_1 = h * y2;
        double k1_2 = h * ((2 * x * y2 - 2 * y1) / (x * x + 1));

        double k2_1 = h * (y2 + 0.5 * k1_2);
        double k2_2 = h * ((2 * (x + 0.5 * h) * (y2 + 0.5 * k1_2) - 2 * (y1 + 0.5 * k1_1)) / ((x + 0.5 * h) * (x + 0.5 * h) + 1));

        double k3_1 = h * (y2 + 0.5 * k2_2);
        double k3_2 = h * ((2 * (x + 0.5 * h) * (y2 + 0.5 * k2_2) - 2 * (y1 + 0.5 * k2_1)) / ((x + 0.5 * h) * (x + 0.5 * h) + 1));

        double k4_1 = h * (y2 + k3_2);
        double k4_2 = h * ((2 * (x + h) * (y2 + k3_2) - 2 * (y1 + k3_1)) / ((x + h) * (x + h) + 1));

        y1 += (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1) / 6;
        y2 += (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2) / 6;
        x += h;

        f_1.push_back(y2);
        f_2.push_back((2 * x * y2 - 2 * y1) / (x * x + 1));
    }

    while (x <= end) {
        x_val.push_back(x);
        y1_val.push_back(y1);
        y2_val.push_back(y2);
        double y1_new = y1 + h / 24 * (55 * f_1.back() - 59 * f_1[f_1.size() - 2] + 37 * f_1[f_1.size() - 3] - 9 * f_1[f_1.size() - 4]);
        double y2_new = y2 + h / 24 * (55 * f_2.back() - 59 * f_2[f_2.size() - 2] + 37 * f_2[f_2.size() - 3] - 9 * f_2[f_2.size() - 4]);
        f_1.push_back(y2_new);
        f_2.push_back((2 * x * y2_new - 2 * y1_new) / (x * x + 1));
        y1 = y1_new;
        y2 = y2_new;
        x += h;
    }
}

int main(){
    double x_0 = 0.0;
    double y1_0 = 1;
    double y2_0 = 1;
    double end = 1.0;
    double h = 0.1;
    vector<double> x_val, y1_val, y2_val;
    vector<double> x_val_half, y1_val_half, y2_val_half;
    euler(h, x_0, y1_0, y2_0, end, x_val, y1_val, y2_val);
    ofstream fout("output.txt");
    fout << "Метод Эйлера:" << endl;
    for (size_t i = 0; i < x_val.size(); i++) {
        double y_exact = exact_solution(x_val[i]);
        fout << "x: " << x_val[i] << ", решение: " << y1_val[i] << ", точное решение: " << y_exact << ", погрешность сравнением с точным решением: " << fabs(y1_val[i] - y_exact) << endl;
    }
    fout << endl;
    euler(h / 2, x_0, y1_0, y2_0, end, x_val_half, y1_val_half, y2_val_half);
    double error = fabs(y1_val_half[y1_val_half.size() - 1] - y1_val[y1_val.size() - 1]) / (pow(2, 4) - 1);
    fout << "Погрешность методом Рунге-Ромберга-Ричардсона для метода Эйлера: " << error << endl;
    fout << endl;
    x_val.clear();
    y1_val.clear();
    y2_val.clear();
    x_val_half.clear();
    y1_val_half.clear();
    y2_val_half.clear();
    runge_kutta(h, x_0, y1_0, y2_0, end, x_val, y1_val, y2_val);
    fout << "Метод Рунге-Кутты:" << endl;
    for (size_t i = 0; i < x_val.size(); i++) {
        double y_exact = exact_solution(x_val[i]);
        fout << "x: " << x_val[i] << ", решение: " << y1_val[i] << ", точное решение: " << y_exact << ", погрешность сравнением с точным решением: " << fabs(y1_val[i] - y_exact) << endl;
    }
    fout << endl;
    runge_kutta(h / 2, x_0, y1_0, y2_0, end, x_val_half, y1_val_half, y2_val_half);
    error = fabs(y1_val_half[y1_val_half.size() - 1] - y1_val[y1_val.size() - 1]) / (pow(2, 4) - 1);
    fout << "Погрешность методом Рунге-Ромберга-Ричардсона для метода Рунге-Кутты: " << error << endl;
    fout << endl;
    x_val.clear();
    y1_val.clear();
    y2_val.clear();
    x_val_half.clear();
    y1_val_half.clear();
    y2_val_half.clear();
    adam(h, x_0, y1_0, y2_0, end, x_val, y1_val, y2_val);
    fout << "Метод Адамса:" << endl;
    for (size_t i = 0; i < x_val.size(); i++) {
        double y_exact = exact_solution(x_val[i]);
        fout << "x: " << x_val[i] << ", решение: " << y1_val[i] << ", точное решение: " << y_exact << ", погрешность сравнением с точным решением: " << fabs(y1_val[i] - y_exact) << endl;
    }
    fout << endl;
    adam(h / 2, x_0, y1_0, y2_0, end, x_val_half, y1_val_half, y2_val_half);
    error = fabs(y1_val_half[y1_val_half.size() - 1] - y1_val[y1_val.size() - 1]) / (pow(2, 4) - 1);
    fout << "Погрешность методом Рунге-Ромберга-Ричардсона для метода Адамса: " << error << endl;
    return 0;
}