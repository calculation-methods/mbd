//
// Created by bigboy on 25.05.2024.
//

#include "runge_kutt.h"
#include <vector>
#include <iostream>
#include "funs.h"


using namespace std;


vector<double> RungeKutt::result(double h, double x0, double y10, double y20, int steps) {
    vector<double> x(steps + 1), y1(steps + 1), y2(steps + 1);
    x[0] = x0; y1[0] = y10; y2[0] = y20;

    for (int i = 0; i < steps; ++i) {
        y1[i + 1] = y1[i] + h * f1(x[i], y1[i], y2[i]);
        y2[i + 1] = y2[i] + h * f2(x[i], y1[i], y2[i]);
        x[i + 1] = x[i] + h;
    }

    cout << "------------Runge-Kutta method------------\n";
    for (int i = 0; i <= steps; ++i) {
        cout << "x: " << x[i] << " y1: " << y1[i] << " y2: " << y2[i] << '\n';
    }
    cout << "Error estimation using the Runge-Romberg method: " << RungeRomberg(y1[steps], y2[steps], 4) << endl;

    return y1;
}