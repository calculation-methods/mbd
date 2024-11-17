#include <vector>
#include <functional>
#include "solve.h"


using namespace std;
double rectangular(function<double(double)> f, double x_start, double x_end, double step){
    double x = x_start, res = 0;
    while (x < x_end){
        res += f((2*x + step)/2);
        x += step;
    }
    return step*res;
}


double trapeze(function<double(double)> f, double x_start, double x_end, double step){
    double x = x_start+step, res = f(x_start)/2 + f(x_end)/2;
    while (x < x_end){
        res += f(x);
        x += step;
    }
    return step * res;
}


double simpson(function<double(double)> f, double x_start, double x_end, double step){
    double x = x_start + step, res = f(x_start) + f(x_end);
    bool flag = true;
    while (x < x_end){
        res += f(x) * ((flag) ? 4 : 2);
        x += step;
        flag = !flag;
    }
    return step * res / 3;
}