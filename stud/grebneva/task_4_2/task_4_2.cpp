#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;

double f(double x, double y, double z){
    return (2*y)/(x*x+1);
}

double accurate_solution(double x){
    return x*x+1;
}

double p(double x){
    return 0;
}

double q(double x){
    return -2;
}

vector<double> runge_kutta(double a, double b, double h, vector<double> x, int n, double y0, double z0) {
    vector<double> y(n);
    vector<double> z(n);

    vector<double> K(4);
    vector<double> L(4);

    y[0] = y0;
    z[0] = z0;

    for (int i = 1; i < n; ++i) {       

        K[0] = h * z[i - 1];
        L[0] = h * f(x[i - 1], y[i - 1], z[i-1]);

        for (int j = 1; j < 4; ++j) {
            K[j] = h * (z[i - 1] + L[j - 1] / 2);
            L[j] = h * f(x[i - 1] + h / 2, y[i - 1] + K[j - 1] / 2, z[i - 1] + L[j - 1] / 2);
        }

        double dy = (K[0] + 2 * K[1] + 2 * K[2] + K[3]) / 6;
        double dz = (L[0] + 2 * L[1] + 2 * L[2] + L[3]) / 6;
        y[i] = y[i - 1] + dy;
        z[i] = z[i - 1] + dz;
    }

    return y;
}

vector<double> shooting_method(double a, double b, double h, vector<double> x, int n, double eps, double y0, double y1){
    double eta0 = 1;
    double eta = 0.8;

    double F0 = runge_kutta(a, b, h, x, n, y0, eta0)[n-1] - y1;
    double F = runge_kutta(a, b, h, x, n, y0, eta)[n-1] - y1;

    while(abs(F) > eps){
        double c = eta;       
        eta = eta - F*(eta - eta0)/(F - F0);
        eta0 = c;
        F0 = F;
        F = runge_kutta(a, b, h, x, n, y0, eta)[n - 1] - y1;
    }
    return runge_kutta(a, b, h, x, n, y0, eta);
}

vector<double> difference_method(double a, double b, double h, vector<double> x, int n, double alpha, double beta, double delta, double gamma, double y0, double y1) { 
    vector<double> A, B, C, D, P(n), Q(n), sol(n);

    A.push_back(0);
    B.push_back(-2 + h * h * q(x[1]));
    C.push_back(1 + p(x[1]) * h / 2);
    D.push_back(-(1 - (p(x[1]) * h) / 2) * y0);
    for (int i = 2; i < n; ++i) {
        
        A.push_back(1 - p(x[i]) * h / 2);
        B.push_back(-2 + h * h * q(x[i]));
        C.push_back(1 + p(x[i]) * h / 2);
        D.push_back(0);
    }
    A.push_back(1 - p(x[n - 2]) * h / 2);
    B.push_back(-2 + h * h * q(x[n - 2]));
    C.push_back(0);
    D.push_back(-(1 + (p(x[n - 2]) * h) / 2) * y1);

    P[0] = (-C[0] / B[0]);
    Q[0] = (D[0] / B[0]);
    for (int i = 1; i <= n; ++i) {
        P[i] = (-C[i] / (B[i] + A[i] * P[i - 1]));
        Q[i] = ((D[i] - A[i] * Q[i - 1]) / (B[i] + A[i] * P[i - 1]));
    }

    sol[n-1] = Q[n-1];
    for (int i = n - 2; i > 0; --i)
        sol[i] = P[i] * sol[i + 1] + Q[i];
    sol[0] = y0;
    sol[n] = y1;
    return sol;
}

pair<vector<double>, vector<double>> RRR_inaccuracy(double a, double b, double h, vector<double> x, int n, double eps, double y0, double y1, double alpha, double beta, double gamma, double delta) {
    vector<double> shooting_method1(n), difference_method1(n);

    double h2 = h/2;
    int n2 = (b - a) / h2 + 1;
    
    vector<double> x2(n);
    for (int i = 0; i < n2; ++i){
        x2[i] = h2 * i;
    }
    vector<double> shooting_method_h = shooting_method(a, b, h, x, n, eps, y0, y1);
    vector<double> shooting_method_h2 = shooting_method(a, b, h2, x2, n2, eps, y0, y1);
    vector<double> difference_method_h = difference_method(a, b, h, x, n, alpha, beta, delta, gamma, y0, y1);
    vector<double> difference_method_h2 = difference_method(a, b, h2, x2, n2, alpha, beta, delta, gamma, y0, y1);

    for (int i = 0; i < n; ++i) {
        shooting_method1[i] = (shooting_method_h2[2 * i] - shooting_method_h[i]) / 15;
        difference_method1[i] = (difference_method_h2[2 * i] - difference_method_h[i]) / 15;
    }
    return make_pair(shooting_method1, difference_method1);
}

int main(){
    double a = 0;
    double b = 2;
    double alpha = 0;
    double beta = 1;
    double delta = -1;
    double gamma = 1;

    double y0 = a*a+1;
    double y1 = b*b+1;
    double h = 0.2;
    double eps = 0.001;

    ofstream fout("answer.txt");
    fout.precision(4);
    fout << fixed;

    int n = 11;
    
    vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i){
        x[i] = h * i + a;
        y[i] = accurate_solution(x[i]);
    }

    vector<double> shoot = shooting_method(a, b, h, x, n, eps, y0, y1);
    fout << "Значения x:" << endl;
    for (int i = 0; i < n; ++i){
        fout << x[i] << " ";
    }

    fout << endl << "Точное решение y:" << endl;
    for (int i = 0; i < n; ++i){
        fout << y[i] << " ";
    }
    fout << endl;

    fout << endl << "Метод shooting:" << endl;
    for (int i = 0; i < n; ++i){
        fout << shoot[i] << " ";
    }

    vector<double> diff = difference_method(a, b, h, x, n, alpha, beta, delta, gamma, y0, y1);

    fout << endl << "Метод difference:" << endl;
    for (int i = 0; i < n; ++i){
        fout << diff[i] << " ";
    }
    fout << endl;

    fout << endl << "Погрешности методом Рунге-Ромберга-Ричардсона" << endl;
    vector<double> r = RRR_inaccuracy(a, b, h, x, n, eps, y0, y1, alpha, beta, gamma, delta).first;

    fout << endl << "Метод shooting погрешность:" << endl;
    for (int i = 0; i < n; ++i){
        fout << r[i] << " ";
    }

    vector<double> rr = RRR_inaccuracy(a, b, h, x, n, eps, y0, y1, alpha, beta, gamma, delta).second;

    fout << endl << "Метод difference погрешность:" << endl;
    for (int i = 0; i < n; ++i){
        fout << rr[i] << " ";
    }
    fout << endl;

    fout << endl << "Погрешности сравнением с точным решением" << endl;

    fout << endl << "Метод shooting погрешность:" << endl;
    for (int i = 0; i < n; ++i){
        fout << abs(shoot[i] - y[i]) << " ";
    }

    fout << endl << "Метод difference погрешность:" << endl;
    for (int i = 0; i < n; ++i){
        fout << abs(diff[i] - y[i]) << " ";
    }

    return 0;
}