#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

using namespace std;

pair<double, vector<double>> Lagrange(vector<double> x, vector<double> y, double xn, int n) {
    double res = 0.0;
    double res_old = 0.0;
    vector<double> w(n, 0);
    for (int i = 0; i < n; i++){
        w[i] = y[i];
        res_old = y[i];
        for (int j = 0; j < n; j++){
            if (i != j) {
                w[i] = w[i]/(x[i] - x[j]);
                res_old = res_old*(xn - x[j])/(x[i] - x[j]);
            }
        }
        res = res + res_old;
    }

    return make_pair(res, w);
}

vector<vector<double>> Newton_Coeffs(vector<double> x, vector<double> y) {
    int n = y.size();
    vector<vector<double>> coefs(n, vector<double>(n));
    for (int j = 0; j < n; ++j) {
        for(int i = n - j - 1; i >= 0; --i) {
            if (j == 0) 
                coefs[i][j] = y[i];
            else {
                coefs[i][j] = (coefs[i][j-1] - coefs[i+1][j-1]) / (x[i] - x[i+j]);
            }
        }
    }
    return coefs;
}


pair<double, vector<double>> Newton(double xn, vector<double> x, int n) {
    vector<double> coefs (n, 0);
    vector<double> y(n);
    
    for(int i = 0; i < n; i++) {
        y[i] = exp(x[i]) + x[i];
    }
    
    vector<vector<double>> coef = Newton_Coeffs(x, y);
    double result = 0;
    
    for(int j = 0; j < n; j++) {
        double res_old = 1;
        for (int i = 0; i < j; i++) {
            res_old *= (xn - x[i]);
        }
        coefs[j] = coef[0][j]; 
        result += res_old * coef[0][j];
    }
    return make_pair(result, coefs);
}


int main() {
    ofstream fout("output.txt");
    int n = 4;
    vector<double> x(n, 0), y(n, 0), w(n, 0);
    double xn = -0.5;
    double res = 0.0;
    x = {-2, -1, 0, 1};
    
    for (int i = 0; i < n; i++){
        y[i] = exp(x[i]) + x[i];
    }
    
    auto [res1,w1] = Lagrange(x, y, xn, n);
    fout << "Коэффициенты Лангранжа:" << endl;
    for (int i = 0; i < n; i++) {
        fout << w1[i] << endl;
    }
    fout << "Значение многочлена Лагранжа: " << res1 << endl;
    fout << "Погрешность интерполяции в точке x: " << abs(res1 - exp(xn) - xn) << endl;
    fout << "----------------------------------------------------" << endl;
    x = {-2, -1, 0.2, 1};
    auto [res2,w2] = Newton(xn, x, n);
    fout << "Коэффициенты Ньютона:" << endl;
    for (int i = 0; i < n; ++i) {
        fout << w2[i] << endl;
    }
    fout << "Значение многочлена Ньютона: " << res2 << endl;
    fout << "Погрешность интерполяции в точке x: " << abs(res2 - exp(xn) - xn) << endl;    

    return 0;
}