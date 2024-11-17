#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

double g(double x) {
    return acos(-0.25*x + 0.5);
}

double f(double x) {
    return cos(x) + 0.25*x - 0.5;
}

double g6(double x) {
    return -sin(x) + 0.25;
}

double h(double x) {
    return -cos(x);
}

pair<double, int> simple_iter(const double q, const double a, const double b, const double eps){

    double x, y, r;
    int k;

    x = (a + b)/2;
    r = 2*eps;
    k = 0;

    while(r > eps){
        y = x;
        x = g(y);
        if ((x < a) || (x > b)) break;
        if (x > y)
            r = x - y;
        else
            r = y - x;
        r = r*q/(1 - q);
        k = k + 1;
    }
    return make_pair(x, k);
}

pair<double, int> newton(const double a, const double b, const double eps){
    double x, y, r;
    int k = 0;

    if(f(a)*h(a) > 0)
        x = a;
    else 
        x = b;   

    r = 2 * eps;
    while(r > eps){
        y = x;
        x = y - f(y)/g6(y);
        if (x > y)
            r = x - y;
        else
            r = y - x;
        k = k + 1;
    }
    return make_pair(x, k);
}

int main() {
    double q, a, b, eps, x;
    int k;

    ifstream fin("input.txt");
    fin >> q;
    fin >> a;
    fin >> b;
    fin >> eps;

    ofstream fout("answer.txt");

    x = simple_iter(q, a, b, eps).first;
    k = simple_iter(q, a, b, eps).second;

    fout << "Метод простой итерации:" << endl;
    fout << "x = " << x << endl;
    fout << "Число итераций: " << k << endl;

    x = newton(a, b, eps).first;
    k = newton(a, b, eps).second;

    fout << "Метод Ньютона:" << endl;
    fout << "x = " << x << endl;
    fout << "Число итераций: " << k << endl;
   
    return 0;
}
