//
// Created by bigboy on 26.05.2024.
//

#ifndef LAB4_2_SHOOTING_H
#define LAB4_2_SHOOTING_H

#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <iomanip>
#include <fstream>
#include "shooting.h"
#include "funcs.h"

class Shooting {
private:
    vector<vector<double>> RungeKutta(function<vector<double>(double, const vector<double>&)> f,
                                      double x0, const vector<double>& Y0, double xf, int N);

    double ShootingFunc(double s, double x_end, int N);
public:
    vector<vector<double>> result(double s_guess, double x_end, int N);
};


#endif //LAB4_2_SHOOTING_H
