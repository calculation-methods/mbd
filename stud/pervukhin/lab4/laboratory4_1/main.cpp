#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

double f1(double x, double y1, double y2) {
    return y2;
}

double f2(double x, double y1, double y2) {
    return ((x + 1) * y2 - y1) / x;
}

double exactSolution(double x) {
    return x + 1 + exp(x);
}

double RungeRomberg(double y_h, double y_h2, int p) {
    return fabs((y_h2 - y_h) / (pow(2, p) - 1));
}

vector<double> Euler(double h, double x0, double y10, double y20, int steps) {
    vector<double> x(steps + 1), y1(steps + 1), y2(steps + 1);
    x[0] = x0; y1[0] = y10; y2[0] = y20;
    ofstream fout("output.txt");
    for (int i = 0; i < steps; ++i) {
        y1[i + 1] = y1[i] + h * f1(x[i], y1[i], y2[i]);
        y2[i + 1] = y2[i] + h * f2(x[i], y1[i], y2[i]);
        x[i + 1] = x[i] + h;
    }

    fout << "------------Метод Эйлера------------\n";
    for (int i = 0; i <= steps; ++i) {
        fout << "x: " << x[i] << " y1: " << y1[i] << " y2: " << y2[i] << '\n';
    }
    fout << "Оценка погрешности методом Рунге Ромберга: " << RungeRomberg(y1[steps], y2[steps], 4) << endl;
    fout.close();
    return y1;
}

vector<double> RungeKutt(double h, double x0, double y10, double y20, int steps) {
    vector<double> x(steps + 1), y1(steps + 1), y2(steps + 1);
    x[0] = x0; y1[0] = y10; y2[0] = y20;
    ofstream fout("output.txt", ios::app);
    for (int i = 0; i < steps; ++i) {
        double k1_y1 = h * f1(x[i], y1[i], y2[i]);
        double k1_y2 = h * f2(x[i], y1[i], y2[i]);

        double k2_y1 = h * f1(x[i] + h/2, y1[i] + k1_y1/2, y2[i] + k1_y2/2);
        double k2_y2 = h * f2(x[i] + h/2, y1[i] + k1_y1/2, y2[i] + k1_y2/2);

        double k3_y1 = h * f1(x[i] + h/2, y1[i] + k2_y1/2, y2[i] + k2_y2/2);
        double k3_y2 = h * f2(x[i] + h/2, y1[i] + k2_y1/2, y2[i] + k2_y2/2);

        double k4_y1 = h * f1(x[i] + h, y1[i] + k3_y1, y2[i] + k3_y2);
        double k4_y2 = h * f2(x[i] + h, y1[i] + k3_y1, y2[i] + k3_y2);

        y1[i + 1] = y1[i] + (k1_y1 + 2*k2_y1 + 2*k3_y1 + k4_y1) / 6;
        y2[i + 1] = y2[i] + (k1_y2 + 2*k2_y2 + 2*k3_y2 + k4_y2) / 6;
        x[i + 1] = x[i] + h;
    }
    fout << endl;
    fout << "------------Метод Рунге - Кутта:------------\n";
    for (int i = 0; i <= steps; ++i) {
        fout << "x: " << x[i] << " y1: " << y1[i] << " y2: " << y2[i] << '\n';
    }
    fout << "Оценка погрешности методом Рунге Ромберга: " << RungeRomberg(y1[steps], y2[steps], 4) << endl;
    fout.close();
    return y1;
}

vector<double> Adams(double h, double x0, double y10, double y20, int steps) {vector<double> x(steps + 1), y1(steps + 1), y2(steps + 1);
    x[0] = x0; y1[0] = y10; y2[0] = y20;
    ofstream fout("output.txt", ios::app);
    
    for (int i = 0; i < 3; ++i) {
        double k1_y1 = h * f1(x[i], y1[i], y2[i]);
        double k1_y2 = h * f2(x[i], y1[i], y2[i]);

        double k2_y1 = h * f1(x[i] + h / 2, y1[i] + k1_y1 / 2, y2[i] + k1_y2 / 2);
        double k2_y2 = h * f2(x[i] + h / 2, y1[i] + k1_y1 / 2, y2[i] + k1_y2 / 2);

        double k3_y1 = h * f1(x[i] + h / 2, y1[i] + k2_y1 / 2, y2[i] + k2_y2 / 2);
        double k3_y2 = h * f2(x[i] + h / 2, y1[i] + k2_y1 / 2, y2[i] + k2_y2 / 2);

        double k4_y1 = h * f1(x[i] + h, y1[i] + k3_y1, y2[i] + k3_y2);
        double k4_y2 = h * f2(x[i] + h, y1[i] + k3_y1, y2[i] + k3_y2);

        y1[i + 1] = y1[i] + (k1_y1 + 2 * k2_y1 + 2 * k3_y1 + k4_y1) / 6;
        y2[i + 1] = y2[i] + (k1_y2 + 2 * k2_y2 + 2 * k3_y2 + k4_y2) / 6;
        x[i+1] = x[i] + h;
    }

    for (int i = 3; i < steps; ++i) {
        double f1i = f1(x[i], y1[i], y2[i]);
        double f2i = f2(x[i], y1[i], y2[i]);

        double f1im1 = f1(x[i] - h, y1[i - 1], y2[i - 1]);
        double f2im1 = f2(x[i] - h, y1[i - 1], y2[i - 1]);

        double f1im2 = f1(x[i] - 2 * h, y1[i - 2], y2[i - 2]);
        double f2im2 = f2(x[i] - 2 * h, y1[i - 2], y2[i - 2]);

        double f1im3 = f1(x[i] - 3 * h, y1[i - 3], y2[i - 3]);
        double f2im3 = f2(x[i] - 3 * h, y1[i - 3], y2[i - 3]);

        y1[i + 1] = y1[i] + (h / 24) * (55 * f1i - 59 * f1im1 + 37 * f1im2 - 9 * f1im3);
        y2[i + 1] = y2[i] + (h / 24) * (55 * f2i - 59 * f2im1 + 37 * f2im2 - 9 * f2im3);
        x[i + 1] = x[i] + h;
    }
    fout << endl;
    fout << "------------Метод Адамса------------" << endl;
    for (int i = 0; i <= steps; ++i) {
        fout << "x: " << x[i] << " y1: " << y1[i] << " y2: " << y2[i] << '\n';
    }
    fout << "Оценка погрешности методом Рунге Ромберга: " << RungeRomberg(y1[steps], y2[steps], 4) << endl;
    fout.close();
    return y1;
}

void Error(const vector<double>& numeric, double x0, double h, int steps) {
    double maxError = 0.0;
    ofstream fout("output.txt", ios::app);
    for (int i = 0; i <= steps; ++i) {
        double exact = exactSolution(x0 + i * h);
        double error = fabs(numeric[i] - exact);
        if (error > maxError) {
            maxError = error;
        }
    }
    fout << "Оценка погрешности методом сравнения: " << maxError << '\n';
}

int main() {
    double h = 0.1;
    double x0 = 1.0;
    double y10 = 2 + exp(1);
    double y20 = 1 + exp(1);
    int steps = (int)((2.0 - 1.0) / h);

    vector<double> eulerResult = Euler(h, x0, y10, y20, steps);
    Error(eulerResult, x0, h, steps);

    vector<double> rungeKuttaResult = RungeKutt(h, x0, y10, y20, steps);
    Error(rungeKuttaResult, x0, h, steps);

    vector<double> adamsResult = Adams(h, x0, y10, y20, steps);
    Error(adamsResult, x0, h, steps);

    return 0;
}
