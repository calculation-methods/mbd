#include <iostream>
#include <cmath>
#include <fstream>

using namespace std;

double F(double x){
    return pow(4,x) - 5 * x - 2;
}

double DF(double x){
    return log(4) * pow(4,x) - 5;
}

double phi(double x){
    return (pow(4,x) - 2)/5;
}

pair <double, int> Newton(double x0, double eps){
    double xn, x;
    x = x0;
    int k = 0;
    for (int i = 0; i < 10000; i++) {

        xn = x - F(x) / DF(x);
        k++;
        if (fabs(xn - x) < eps)
            break;
        x = xn;
    }
    return make_pair(xn, k);
}

pair <double, int> SimpleIterations(double x0, double eps){
    double xn, x;
    x = x0;
    int k = 0;
    for (int i = 0; i < 10000; i++) {
        xn = phi(x);
        k++;
        if (fabs(xn - x) < eps)
            break;
        x = xn;
    }
    return make_pair(xn, k);
}

int main() {
    ofstream fout;
    fout.open("output.txt");
    double x0 = -0.3;
    double x1 = 1.5;
    double eps = 0.0001;
    auto [newton, k] = Newton(x0, eps);
    auto [simpIt, n] = SimpleIterations(x0, eps);
    fout << "Начальное приближение x_0 = "<< x0 << endl;
    fout << "Решение методом Ньютона:" << newton << " | Количество итераций: " << k << endl;
    fout << "Решение методом простых итераций:" << simpIt << " | Количество итераций: " << n << endl;
    fout << "----------------------------------------------------------------------" << endl;
    auto [newton1, k1] = Newton(x1, eps);
    auto [simpIt1, n1] = SimpleIterations(x1, eps);
    fout << "Начальное приближение x_0 = "<< x1 << endl;
    fout << "Решение методом Ньютона:" << newton1 << " | Количество итераций: " << k1 << endl;
    fout << "Решение методом простых итераций:" << simpIt1 << " | Количество итераций: " << n1 << endl;

    return 0;
}
