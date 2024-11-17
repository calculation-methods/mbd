//
// Created by bigboy on 26.05.2024.
//

#include "funcs.h"
#include <cmath>
#include <vector>
#include <iostream>


using namespace std;

double f(double x) {
      return x*x + x + 1 + (x*x + 1) * atan(x);

}

vector<double> odes(double x, const vector<double>& Y) {
    double y = Y[0];
    double dy = Y[1];
    double d2y = (2 * y) / (x * x + 1);

    return {dy, d2y};
}


double rungeRomberg(const vector<double>& y2h, const vector<double>& yh, int N) {
    double error = 0.0;
    for (int i = 0; i <= N; ++i) {
        error = max(error, abs(y2h[2 * i] - yh[i]) / 3.0);
    }
    return error;
}