#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;

double Y(double x){
    return 1/(256 - pow(x,4));
}

pair <double, double> Rectangle(double x0, double xk, double h1, double h2){
    int n1 = 0, n2 = 0;
    double x = x0;
    double x1 = x0;
    double res = 0, res1 = 0;
    double h = 0;
    vector <double> xi1, xi2;
    while (x != xk){
        xi1.push_back(x);
        x += h1;
        n1++;
    }
    n1++;
    xi1.push_back(x);
    while (x1 != xk){
        xi2.push_back(x1);
        x1 += h2;
        n2++;
    }
    n2++;
    xi2.push_back(x1);

    for (int i = 1; i <= n1; i++){
        h = xi1[i] - xi1[i-1];
        res += h * Y((xi1[i] + xi1[i-1]) / 2);
    }

    for (int i = 1; i <= n2; i++){
        h = xi2[i] - xi2[i-1];
        res1 += h * Y((xi2[i] + xi2[i-1]) / 2);
    }
    return make_pair(res, res1);
}

pair <double, double> Trapez(double x0, double xk, double h1, double h2){
    int n1 = 0, n2 = 0;
    double x = x0;
    double x1 = x0;
    double res = 0, res1 = 0;
    double h = 0;
    vector <double> xi1, xi2;
    while (x != xk){
        xi1.push_back(x);
        x += h1;
        n1++;
    }
    n1++;
    xi1.push_back(x);
    while (x1 != xk){
        xi2.push_back(x1);
        x1 += h2;
        n2++;
    }
    n2++;
    xi2.push_back(x1);

    for (int i = 1; i <= n1; i++){
        h = xi1[i] - xi1[i-1];
        res += h * (Y(xi1[i]) + Y(xi1[i-1]));
    }
    res *= 0.5;
    for (int i = 1; i <= n2; i++){
        h = xi2[i] - xi2[i-1];
        res1 += h * (Y(xi2[i]) + Y(xi2[i-1]));
    }
    res1 *= 0.5;
    return make_pair(res, res1);
}

pair <double, double> Simpson(double x0, double xk, double h1, double h2){
    int n1 = 0, n2 = 0;
    double x = x0;
    double x1 = x0;
    double res = 0, res1 = 0;
    double h = 0;
    vector <double> xi1, xi2;
    while (x != xk){
        xi1.push_back(x);
        x += h1;
        n1++;
    }
    n1++;
    xi1.push_back(x);
    while (x1 != xk){
        xi2.push_back(x1);
        x1 += h2;
        n2++;
    }
    n2++;
    xi2.push_back(x1);

    for (int i = 1; i <= n1; i++){
        h = (xi1[i] - xi1[i-1]) / 2;
        res += h * (Y(xi1[i-1]) + 4 * Y((xi1[i-1] + xi1[i])/2) + Y(xi1[i]));
    }
    res  = res / 3;
    for (int i = 1; i <= n2; i++){
        h = (xi2[i] - xi2[i-1]) / 2;
        res1 += h * (Y(xi2[i-1]) + 4 * Y((xi2[i-1] + xi2[i])/2) + Y(xi2[i]));;
    }
    res1  = res1 / 3;
    return make_pair(res, res1);
}

double Runge(double F1, double F2, double p){
    return F1 + (F1 - F2)/(pow(2, p) - 1);
}

int main() {
    ofstream fout("output.txt");
    double x0 = -2;
    double xk = 2;
    double h1 = 1;
    double h2 = 0.5;
    double res1 = Rectangle(x0, xk, h1, h2).first, res2 = Rectangle(x0, xk, h1, h2).second;
    fout << "---------- Метод прямоугольников ----------" << endl;
    fout << "Результат с шагом " << h1 << ": " << res1 << endl << "Результат с шагом " << h2 << ": " << res2 << endl;
    fout << "Погрешность: " << Runge(res1, res2, 2) << endl;
    fout << "---------- Метод трапеций ----------" << endl;
    res1 = Trapez(x0, xk, h1, h2).first;
    res2 = Trapez(x0, xk, h1, h2).second;
    fout << "Результат с шагом " << h1 << ": " << res1 << endl << "Результат с шагом " << h2 << ": " << res2 << endl;
    fout << "Погрешность: " << Runge(res1, res2, 2) << endl;
    fout << "---------- Метод Симпсона ----------" << endl;
    res1 = Simpson(x0, xk, h1, h2).first;
    res2 = Simpson(x0, xk, h1, h2).second;
    fout << "Результат с шагом " << h1 << ": " << res1 << endl << "Результат с шагом " << h2 << ": " << res2 << endl;
    fout << "Погрешность: " << Runge(res1, res2, 2) << endl;
    return 0;
}
