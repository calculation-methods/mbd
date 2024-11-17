#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

double f(double x) {
    return log(x);
}

pair<vector<double>, double> lagrange(const int n, vector<double> x, vector<double> y, double starX) {
    double v = 0.0, func = 0.0;
    vector<double> w(n, 0);

    for (int j = 0; j < n; ++j){
        w[j] = y[j];
        func = y[j];
        for (int i = 0; i < n; ++i){
            if (i != j) {
                w[j] = w[j]/(x[j] - x[i]);
                func = func*(starX - x[i])/(x[j] - x[i]);
            }
        }
        v = v + func;
    }

    return make_pair(w, v);
}

pair<vector<double>, double> newton(const int n, vector<double> x, vector<double> y, double starX) {
    double v = 0.0, func = 0.0;
    vector<double> w(n, 0);
    vector<vector<double>> d(n, vector<double>(n, 0));

    for (int i = 0; i < n; ++i){
        d[i][0] = y[i];
    }
    for (int j = 1; j < n; j++){
        for (int i = 0; i < (n - j); i++){
            d[i][j] = (d[i][j - 1] - d[i + 1][j - 1])/(x[i] - x[i + j]);
        }
    }
    for (int j = 0; j < n; ++j){
        w[j] = d[0][j];
        func = d[0][j];
        for (int i = 0; i < j; ++i){
            func = func * (starX - x[i]);
        }
        v = v + func;
    }

    return make_pair(w, v);
}

int main() {
    int n = 4;
    vector<double> x(n, 0), y(n, 0), w(n, 0);
    double starX = 0.0, v = 0.0;

    ifstream fin("input.txt");
    for (int i = 0; i < n; ++i){
        fin >> x[i];
        y[i] = f(x[i]);
    }
    fin >> starX;

    w = lagrange(n, x, y, starX).first;
    v = lagrange(n, x, y, starX).second;

    ofstream fout("answer.txt");
    fout.precision(4);
    fout << fixed;

    fout << "Коэффициенты:" << endl;
    for (int i = 0; i < n; ++i) {
        fout << w[i] << endl;
    }

    double delta = 0.0;
    delta = abs(v - f(starX));

    fout << "Значение многочлена Лагранжа в точке x: " << v << endl;
    fout << "Абсолютная погрешность интерполяции в точке x: " << delta << endl;

    w = newton(n, x, y, starX).first;
    v = newton(n, x, y, starX).second;

    fout << endl << "Коэффициенты:" << endl;
    for (int i = 0; i < n; ++i) {
        fout << w[i] << endl;
    }

    delta = abs(v - f(starX));

    fout << "Значение многочлена Ньютона в точке x: " << v << endl;
    fout << "Абсолютная погрешность интерполяции в точке x: " << delta << endl;    

    return 0;
}