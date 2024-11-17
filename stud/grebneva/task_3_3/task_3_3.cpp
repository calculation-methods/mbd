#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

void MNK(vector<double> x, vector<double> y, const int n, const int m, vector<double>& b, vector<double>& z, vector<vector<double>>& a, double& f){
    double g = 0.0;
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < m; ++j){
            a[i][j] = 0;
            for (int k = 0; k < n; ++k){
                a[i][j] = a[i][j] + pow(x[k], i+j);
            }
        }
        b[i] = 0;
        for (int k = 0; k < n; ++k){
            b[i] = b[i]+y[k]*pow(x[k], i);
        }
    }
    if (m == 1) z[0] = b[0]/a[0][0];
    if (m == 2) {
        g = a[0][0] * a[1][1] - a[1][0] * a[0][1];
        z[0] = (b[0]*a[1][1] - b[1]*a[1][0])/g;
        z[1] = (a[0][0]*b[1] - a[1][0]*b[0])/g;
    }
    if (m == 3) {
        g = a[0][0]*a[1][1]*a[2][2] + a[1][0]*a[2][1]*a[0][2] + a[2][0]*a[0][1]*a[1][2] - a[2][0]*a[1][1]*a[0][2] - a[0][0]*a[2][1]*a[1][2] - a[1][0]*a[0][1]*a[2][2];
        z[0] = (b[0]*a[1][1]*a[2][2] + b[1]*a[2][1]*a[0][2] + b[2]*a[0][1]*a[1][2] - b[2]*a[1][1]*a[0][2] - b[0]*a[2][1]*a[1][2] - b[1]*a[0][1]*a[2][2])/g;
        z[1] = (a[0][0]*b[1]*a[2][2] + a[1][0]*b[2]*a[0][2] + a[2][0]*b[0]*a[1][2] - a[2][0]*b[1]*a[0][2] - a[0][0]*b[2]*a[1][2] - a[1][0]*b[0]*a[2][2])/g;
        z[2] = (a[0][0]*a[1][1]*b[2] + a[1][0]*a[2][1]*b[0] + a[2][0]*a[0][1]*b[1] - a[2][0]*a[1][1]*b[0] - a[0][0]*a[2][1]*b[1] - a[1][0]*a[0][1]*b[2])/g;
    }

    for (int k = 0; k < n; ++k) {
        g = 0;
        for (int i = 0; i < m; ++i) {
            g = g + z[i]*pow(x[k], i);
        }
        f = f + (g - y[k])*(g - y[k]);
    }
}

int main() {
    int n = 6, m = 2;
    vector<double> x(n, 0), y(n, 0), b(m, 0), z(m, 0);
    vector<vector<double>> a(m, vector<double>(m, 0));
    double f = 0.;

    ifstream fin("input.txt");
    for (int i = 0; i < n; ++i){
        fin >> x[i];     
    }
    for (int i = 0; i < n; ++i){
        fin >> y[i];     
    }

    ofstream fout("answer.txt");
    fout.precision(4);
    fout << fixed;

    MNK(x, y, n, m, b, z, a, f);

    fout << "1-ая СТЕПЕНЬ" << endl;
    fout << "Нормальная система метода наименьших квадратов:" << endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j){
            fout << a[i][j] << "*a" << j;
            if (j != m-1) fout << " + ";
            else fout << " = ";
        }
        fout << b[i] << endl;
    }

    fout << "Коэффициенты приближающего многочлена:" << endl;
    for (int i = 0; i < m; ++i) {
        fout << z[i] << " ";
    }
    fout << endl;

    fout << "Значение суммы квадратов ошибок: " << f << endl;

    m = 3;
    vector<double> b1(m, 0), z1(m, 0);
    vector<vector<double>> a1(m, vector<double>(m, 0));
    f = 0.;
    MNK(x, y, n, m, b1, z1, a1, f);

    fout << endl << "2-ая СТЕПЕНЬ" << endl;
    fout << "Нормальная система метода наименьших квадратов:" << endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j){
            fout << a1[i][j] << "*a" << j;
            if (j != m-1) fout << " + ";
            else fout << " = ";
        }
        fout << b1[i] << endl;
    }

    fout << "Коэффициенты приближающего многочлена:" << endl;
    for (int i = 0; i < m; ++i) {
        fout << z1[i] << " ";
    }
    fout << endl;

    fout << "Значение суммы квадратов ошибок: " << f << endl;

    return 0;
}