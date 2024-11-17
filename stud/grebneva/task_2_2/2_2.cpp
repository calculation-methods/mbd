#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

double f(double x1, double x2) {
    return cos(x2) + 1;
}

double g(double x1, double x2) {
    return log10(x1+1) + 2;
}

double f1(double x1, double x2) {
    return x1 - cos(x2) - 1;
}

double g1(double x1, double x2) {
    return x2 - log10(x1+1) - 2;
}

double f1_x1(double x1, double x2) {
    return 1;
}

double g1_x1(double x1, double x2) {
    return -1/(log(10)*(x1+1));
}

double f1_x2(double x1, double x2) {
    return sin(x2);
}

double g1_x2(double x1, double x2) {
    return 1;
}


pair<double, double> simple_iter(const double q, const double a, const double b, const double c, const double d, const double eps, int &k){

    double x1, x2, u, v, r, s;

    x1 = (a + b)/2;
    x2 = (c + d)/2;
    r = 2*eps;
    k = 0;

    while(r > eps){
        u = x1;
        v = x2;
        x1 = f(u, v);
        x2 = g(u, v);
        //if ((x1 < a) || (x1 > b) || (x2 < c) || (x2 > d)) exit(0);
        if (x1 > u)
            r = x1 - u;
        else
            r = u - x1;

        if (x2 > v)
            s = x2 - v;
        else
            s = v - x2;

        if (s > r) 
            r = s;

        r = r*q/(1 - q);
        k = k + 1;
    }
    return make_pair(x1, x2);
}

pair<double, double> newton(const double a, const double b, const double c, const double d, const double eps, int &k){
    double x1, x2, u, v, r, s;

    x1 = (a + b)/2;
    x2 = (c + d)/2;
    r = 2*eps;
    k = 0;

    while(r > eps){
        u = x1;
        v = x2;
        s =f1_x1(u, v)*g1_x2(u, v) - f1_x2(u, v)*g1_x1(u,v);
        //if (s == 0) exit(0);
        x1 = u - (f1(u, v)*g1_x2(u, v) - g1(u, v)*f1_x2(u, v))/s;
        x2 = v - (f1_x1(u, v)*g1(u, v) - g1_x1(u, v)*f1(u, v))/s;
        if (x1 > u)
            r = x1 - u;
        else
            r = u - x1;

        if (x2 > v)
            s = x2 - v;
        else
            s = v - x2;

        if (s > r) 
            r = s;

        k = k + 1;
    }
    return make_pair(x1, x2);
}

int main() {
    double q, a, b, c, d, eps, x1, x2;
    int k;

    ifstream fin("input.txt");
    fin >> q;
    fin >> a;
    fin >> b;
    fin >> c;
    fin >> d;
    fin >> eps;

    ofstream fout("answer.txt");

    x1 = simple_iter(q, a, b, c, d, eps, k).first;
    x2 = simple_iter(q, a, b, c, d, eps, k).second;

    fout << "Метод простой итерации:" << endl;
    fout << "x1 = " << x1 << endl;
    fout << "x2 = " << x2 << endl;
    fout << "Число итераций: " << k << endl;

    x1 = newton(a, b, c, d, eps, k).first;
    x2 = newton(a, b, c, d, eps, k).second;

    fout << "Метод Ньютона:" << endl;
    fout << "x1 = " << x1 << endl;
    fout << "x2 = " << x2 << endl;
    fout << "Число итераций: " << k << endl;
   
    return 0;
}
