//
// Created by bigboy on 26.05.2024.
//

#ifndef LAB4_2_FUNCS_H
#define LAB4_2_FUNCS_H
#include <vector>
using namespace std;

double f(double x);

vector<double> odes(double x, const vector<double>& Y);
double rungeRomberg(const vector<double>& y2h, const vector<double>& yh, int N);

#endif //LAB4_2_FUNCS_H
