//
// Created by bigboy on 25.05.2024.
//
#include <cmath>

#include "funs.h"
double f1(double x, double y1, double y2) {
    return (x*x  * y2 - y1)/(x+1);
}

double f2(double x, double y1, double y2) {
    return (y1 - (x+1) * y2)/(x*x);
}

double F(double x) {
    return x + 1 + x*exp(1/x);
}

double RungeRomberg(double y_h, double y_h2, int p) {
    return fabs((y_h2 - y_h) / (pow(2, p) - 1));
}


