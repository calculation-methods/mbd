#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;

double f(double x, double y, double z){
    return (x*z-y)/(x*x-x);
}

double exact_solution(double x){
    return 2+x+2*x*log(fabs(x));
}

double p(double x){
    return -x;
}

double q(double x){
    return 1;
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

vector<double> difference_method(double a, double b, double h, vector<double> x, int n, double y0, double y1) { 
    vector<double> A, B, C, D, P(n), Q(n), res(n);

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

    res[n-1] = Q[n-1];
    for (int i = n - 2; i > 0; --i)
        res[i] = P[i] * res[i + 1] + Q[i];
    res[0] = y0;
    res[n] = y1;
    return res;
}

int main(){
    double a = 1.0001;
    double b = 3;

    double y0 = 2+a+2*a*log(a);
    double y1 = 2+b+2*b*log(b);
    double h = 0.2;
    double eps = 0.0001;

    ofstream fout("output.txt");

    int n = 11;

    vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i){
        x[i] = h * i + a;
        y[i] = exact_solution(x[i]);
    }

    vector<double> shoot = shooting_method(a, b, h, x, n, eps, y0, y1);

    fout << endl << "Точное решение:" << endl;
    for (int i = 0; i < n; ++i){
        fout << y[i] << " ";
    }
    fout << endl;
    fout << endl << "Метод стрельбы:" << endl;
    for (int i = 0; i < n; ++i){
        fout << shoot[i] << " ";
    }
    vector<double> diff = difference_method(a, b, h, x, n, y0, y1);

    fout << endl << "Метод difference:" << endl;
    for (int i = 0; i < n; ++i){
        fout << diff[i] << " ";
    }
    fout << endl;

    fout << endl << "Погрешности методом Рунге-Ромберга-Ричардсона" << endl;
    vector<double> shoot_half = shooting_method(a, b, h/2, x, n, eps, y0, y1);
    double error = fabs(shoot_half[shoot_half.size() - 1] - shoot[shoot.size() - 1]) / (pow(2, 4) - 1);
    fout << "Метод стрельбы: " << error << endl;

    vector<double> diff_half = difference_method(a, b, h/2, x, n, y0, y1);
    error = fabs(diff_half[diff_half.size() - 1] - diff[diff.size() - 1]) / (pow(2, 4) - 1);

    fout << "Метод difference: " << error << endl;
    fout << endl;

    fout << "Погрешности сравнением с точным решением" << endl;
    fout << "Метод стрельбы:" << endl;
    for (int i = 0; i < n; ++i){
        fout << abs(shoot[i] - y[i]) << " ";
    }
    fout << endl << "Метод difference:" << endl;
    for (int i = 0; i < n; ++i){
        fout << abs(diff[i] - y[i]) << " ";
    }
    return 0;
}