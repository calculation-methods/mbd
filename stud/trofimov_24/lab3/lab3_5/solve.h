//
// Created by владислав трофимов on 11.05.2024.
//

#ifndef INC_3_5_SOLVE_H
#define INC_3_5_SOLVE_H
#include <vector>
#include <functional>


using namespace std;
double rectangular(function<double(double)> f, double x_start, double x_end, double step);

double trapeze(function<double(double)> f, double x_start, double x_end, double step);
double simpson(function<double(double)> f, double x_start, double x_end, double step);


#endif //INC_3_5_SOLVE_H
