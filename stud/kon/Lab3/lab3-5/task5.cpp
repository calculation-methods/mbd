#include <iostream>
#include <cmath>
#include <fstream>

using namespace std;

double y(double x) {
    return x / (pow(x, 3) + 8);
}

double rectangle(double x_0, double x_k, double h) {
    double res = 0.0;
    for (double x = x_0; x < x_k; x += h) {
        res += y(x) * h;
    }
    return res;
}

double trapezoid(double x_0, double x_k, double h) {
    double res = 0.0;
    for (double x = x_0; x < x_k; x += h) {
        res += (y(x) + y(x + h)) * h / 2.0;
    }
    return res;
}

double simpson(double x_0, double x_k, double h) {
    double res = 0.0;
    for (double x = x_0; x < x_k; x += 2 * h) {
        res += (y(x) + 4 * y(x + h) + y(x + 2 * h)) * h / 3.0;
    }
    return res;
}

double runge_romberg(double I1, double I2, double p) {
    return (I2 - I1) / (pow(2, p) - 1);
}

int main(){
    double x_0 = -1.0, x_k = 1.0, h1 = 0.5, h2 = 0.25;
    ofstream fout("output.txt");
    double rectangle_res_h1 = rectangle(x_0, x_k, h1);
    fout << "Метод прямоугольников с шагом h1: " << rectangle_res_h1 << endl;
    double rectangle_res_h2 = rectangle(x_0, x_k, h2);
    fout << "Метод прямоугольников с шагом h2: " << rectangle_res_h2 << endl;
    double delta_rectangle = runge_romberg(rectangle_res_h1, rectangle_res_h2, 2);
    fout << "Погрешность: " << delta_rectangle << endl;
    fout << endl;

    double trapezoid_res_h1 = trapezoid(x_0, x_k, h1);
    fout << "Метод трапеций с шагом h1: " << trapezoid_res_h1 << endl;
    double trapezoid_res_h2 = trapezoid(x_0, x_k, h2);
    fout << "Метод трапеций с шагом h2: " << trapezoid_res_h2 << endl;
    double delta_trapezoid = runge_romberg(trapezoid_res_h1, trapezoid_res_h2, 2);
    fout << "Погрешность: " << delta_trapezoid << endl;
    fout << endl;

    double simpson_res_h1 = simpson(x_0, x_k, h1);
    fout << "Метод Симпсона с шагом h1: " << simpson_res_h1 << endl;
    double simpson_res_h2 = simpson(x_0, x_k, h2);
    fout << "Метод Симпсона с шагом h2: " << simpson_res_h2 << endl;
    double delta_simpson = runge_romberg(simpson_res_h1, simpson_res_h2, 4);
    fout << "Погрешность: " << delta_simpson << endl;
    return 0;
}