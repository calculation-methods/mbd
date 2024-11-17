#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

double splain(const int n, vector<double> x, vector<double> y, double starX, vector<double>& a, vector<double>& b, vector<double>& c, vector<double>& d){
    vector<double> p(n-2, 0), q(n-2, 0);
    double v = 0.0;
    for (int i = 0; i < n - 1; ++i){
        a[i] = y[i];
    }
    c[0] = 0;
    p[0] = -(x[2] - x[1])/(2*x[2] - 2*x[0]);
    q[0] = 3*((y[2] - y[1])/(x[2] - x[1]) - (y[1] - y[0])/(x[1] - x[0]))/(2*x[2] - 2*x[0]);
    for (int i = 1; i < n-2; ++i){
        p[i] = -(x[i+2] - x[i+1])/((2*x[i+2] - 2*x[i]) + (x[i+1] - x[i])*p[i-1]);
        q[i] = (3*((y[i+2] - y[i+1])/(x[i+2] - x[i+1]) - (y[i+1] - y[i])/(x[i+1] - x[i])) - (x[i+1] - x[i])*q[i-1])/((2*x[i+2] - 2*x[i]) + (x[i+1] - x[i])*p[i-1]);
    }
    c[n-2] = q[n-3];
    for (int i = n-3; i >= 1; --i){
        c[i] = p[i-1]*c[i+1] + q[i-1];
    }
    for (int i = 0; i < n-2; ++i){
        b[i] = (y[i+1] - y[i])/(x[i+1] - x[i]) - (x[i+1] - x[i])*(c[i+1] + 2*c[i])/3;
        d[i] = (c[i+1] - c[i])/(x[i+1] - x[i])/3;
    }
    b[n-2] = (y[n-1] - y[n-2])/(x[n-1] - x[n-2]) - (x[n-1] - x[n-2])*2*c[n-2]/3;
    d[n-2] = -c[n-2]/(x[n-1] - x[n-2])/3;
    int j=0;
    for (int i = 0; i < n-1; ++i){
        if ((starX >= x[i]) && (starX <= x[i+1])) j = i;
    }
    if (j == 0) return 0;
    starX = starX - x[j];
    v = a[j] + b[j]*starX + c[j]*starX*starX + d[j]*starX*starX*starX;
    return v;
}

int main() {
    int n = 5;
    vector<double> x(n, 0), y(n, 0), w(n, 0);
    double starX = 0.0;
    vector<double> a(n-1, 0), b(n-1, 0), c(n-1, 0), d(n-1, 0);

    ifstream fin("input.txt");
    for (int i = 0; i < n; ++i){
        fin >> x[i];     
    }
    for (int i = 0; i < n; ++i){
        fin >> y[i];     
    }
    fin >> starX;

    ofstream fout("answer.txt");
    fout.precision(4);
    fout << fixed;

    double v = splain(n, x, y, starX, a, b, c, d);

    fout << "Коэффициенты:" << endl;
    for (int i = 0; i < n-1; ++i) {
        fout << a[i] << " " << b[i] << " " << c[i] << " " << d[i] << endl;
    }

    fout << "Значение функции в точке X*: " << v << endl;

    return 0;
}