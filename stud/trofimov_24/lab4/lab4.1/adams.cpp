//
// Created by bigboy on 25.05.2024.
//

#include "adams.h"
#include "funs.h"
#include <iostream>


using namespace std;

vector<double> Adams::result(double h, double x0, double y10, double y20, int steps){
    vector<double> x(steps + 1), y1(steps + 1), y2(steps + 1);
    x[0] = x0; y1[0] = y10; y2[0] = y20;

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
    cout << endl;
    cout << "------------Adams method------------" << endl;
    for (int i = 0; i <= steps; ++i) {
        cout << "x: " << x[i] << " y1: " << y1[i] << " y2: " << y2[i] << '\n';
    }
    cout << "Error estimation using the Runge-Romberg method:  " << RungeRomberg(y1[steps], y2[steps], 4) << endl;
    return y1;
}
