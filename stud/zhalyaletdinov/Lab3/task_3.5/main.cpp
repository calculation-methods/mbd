#include <bits/stdc++.h>
using namespace std;

double rectangular(function<double(double)> f, double x_begin, double x_end, double k){
    double x = x_begin, res = 0;
    while (x < x_end){
        res += f((2*x + k)/2);
        x += k;
    }
    return k*res;
}


double trapeze(function<double(double)> f, double x_begin, double x_end, double k){
    double x = x_begin+k, res = f(x_begin)/2 + f(x_end)/2;
    while (x < x_end){
        res += f(x);
        x += k;
    }
    return k * res;
}


double simpson(function<double(double)> f, double x_begin, double x_end, double k){
    double x = x_begin + k, res = f(x_begin) + f(x_end);
    bool flag = true;
    while (x < x_end){
        if (flag)
            res += f(x) * 4;
        else
            res += f(x) * 2;
        x += k;
        flag = !flag;
    }
    return k * res / 3;
}

double Runge(double v1, double v2, double k1, double k2, double p){
    return v1 + (v1 - v2)/(pow((k2/k1), p) - 1);
}

int main() {
    auto y = [](double x) { 
        return 1 / (pow(x, 2) + 4); 
    };
    double x0 = -2, xk = 2, val1, val2;

    cout << "Метод прямоугольников" << endl;
    val1 = rectangular(y, x0, xk, 1.0);
    val2 = rectangular(y, x0, xk, 0.5);
    cout << "Значение интеграла с шагом 1.0 " << val1 << endl;
    cout << "Значение интеграла с шагом 0.5 " << val1 << endl;
    cout << "\tЗначение Рунге-Ромберга " << Runge(val1, val2, 1.0, 0.5, 2) << endl;

    cout << "Метод трапеций" << endl;
    val1 = trapeze(y, x0, xk, 1.0);
    val2 = trapeze(y, x0, xk, 0.5);
    cout << "Значение интеграла с шагом 1.0 " << val1 << endl;
    cout << "Значение интеграла с шагом 0.5 " << val1 << endl;
    cout << "\tЗначение Рунге-Ромберга " << Runge(val1, val2, 1.0, 0.5, 2) << endl;

    cout << "Метод Симпсона" << endl;
    val1 = simpson(y, x0, xk, 1.0);
    val2 = simpson(y, x0, xk, 0.5);
    cout << "Значение интеграла с шагом 1.0 " << val1 << endl;
    cout << "Значение интеграла с шагом 0.5 " << val1 << endl;
    cout << "\tЗначение Рунге-Ромберга " << Runge(val1, val2, 1.0, 0.5, 2) << endl;

    return 0;
}