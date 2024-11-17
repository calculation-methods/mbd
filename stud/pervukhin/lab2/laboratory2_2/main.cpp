#include <iostream>
#include <cmath>
#include <fstream>

using namespace std;

double F1(double x1, double x2){
    return 3*x1 - cos(x2);
}

double F2(double x1, double x2){
    return 3*x2 - exp(x1);
}

double DF1_dx1(double x1, double x2){
    return 3;
}

double DF2_dx1(double x1, double x2){
    return -exp(x2);
}

double DF1_dx2(double x1, double x2){
    return sin(x2);
}

double DF2_dx2(double x1, double x2){
    return 3;
}

double Phi1(double x1, double x2){
    return cos(x2)/3;
}

double Phi2(double x1, double x2){
    return exp(x1)/3;
}

tuple <double, double, int> SimpleIterations(double x1_0, double x2_0, double eps){
    double x1 = x1_0, x2 = x2_0;
    double xn1, xn2;
    int k = 0;
    for (int i = 1; i < 1000; i++){
        xn1 = Phi1(x1, x2);
        xn2 = Phi2(x1,x2);
        k++;
        if (max(fabs(xn1 - x1), fabs(xn2 - x2)) < eps){
            break;
        }
        x1 = xn1;
        x2 = xn2;
    }
    return make_tuple(xn1, xn2, k);
}

tuple <double, double, int> Newton(double x1_0, double x2_0, double eps){
    double x1 = x1_0, x2 = x2_0;
    double xn1, xn2;
    double detA1, detA2, detJ;
    int k = 0;
    for (int i = 0; i < 1000; i++){
        k++;
        detA1 = F1(x1, x2) * DF2_dx2(x1, x2) - F2(x1, x2) * DF1_dx2(x1, x2);
        detA2 = DF1_dx1(x1, x2) * F2(x1, x2) - DF2_dx1(x1, x2) * F1(x1, x2);
        detJ = DF1_dx1(x1, x2) * DF2_dx2(x1, x2) - DF2_dx1(x1, x2) * DF1_dx2(x1, x2);
        xn1 = x1 - detA1 / detJ;
        xn2 = x2 - detA2 / detJ;
        if (max(fabs(xn1 - x1), fabs(xn2 - x2)) < eps){
            break;
        }
        x1 = xn1;
        x2 = xn2;
    }
    return make_tuple(x1, x2, k);
}

int main() {
    ofstream fout;
    fout.open("output.txt");
    double x1 = 0.5, x2 = 0.5;
    double eps = 0.0001;
    auto [res1, res2, iter] = SimpleIterations(x1, x2, eps);
    fout << "Решение методом простых итераций:" << endl;
    fout << "x1 = " << res1 << " | x2 = " << res2 << endl;
    fout << "Количество итераций:" << iter << endl;
    fout << "-----------------------------------" << endl;
    auto [res12, res22, iter2] = Newton(x1, x2, eps);
    fout << "Решение методом Ньютона:" << endl;
    fout << "x1 = " << res12 << " | x2 = " << res22 << endl;
    fout << "Количество итераций:" << iter2 << endl;
    return 0;
}
