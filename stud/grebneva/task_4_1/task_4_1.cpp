#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;

double f(double x, double y){
    return (1 + 2*tan(x)*tan(x))*y;
}

double accurate_solution(double x){
    return 1/cos(x) + sin(x) + x/cos(x);
}

vector<double> euler(double a, double b, double h, vector<double> x, int n){
    vector<double> y(n, 1.0);
    vector<double> z(n, 2.0);

    for (int i = 1; i < n; ++i) {
        y[i] = y[i - 1] + h * z[i - 1];
        z[i] = z[i - 1] + h * f(x[i - 1], y[i - 1]);
    }

    return y;
}

pair<vector<double>, vector<double>> runge_kutta(double a, double b, double h, vector<double> x, int n) {
    vector<double> y(n, 1.0);
    vector<double> z(n, 2.0);

    vector<double> K(4);
    vector<double> L(4);

    for (int i = 1; i < n; ++i) {       

        K[0] = h * z[i - 1];
        L[0] = h * f(x[i - 1], y[i - 1]);

        for (int j = 1; j < 4; ++j) {
            K[j] = h * (z[i - 1] + L[j - 1] / 2);
            L[j] = h * f(x[i - 1] + h / 2, y[i - 1] + K[j - 1] / 2);
        }

        double dy = (K[0] + 2 * K[1] + 2 * K[2] + K[3]) / 6;
        double dz = (L[0] + 2 * L[1] + 2 * L[2] + L[3]) / 6;
        y[i] = y[i - 1] + dy;
        z[i] = z[i - 1] + dz;
    }

    return make_pair(y, z);
}

vector<double> adams(double a, double b, double h, vector<double> x, int n) {
    vector<double> y(n, 0.0);
    vector<double> z(n, 0.0);

    vector<double> y0 = runge_kutta(a, a + 3 * h, h, x, n).first;
    for (int i = 0; i < y0.size(); ++i) {
        y[i] = y0[i];
    }

    vector<double> z0 = runge_kutta(a, a + 3 * h, h, x, n).second;
    for (int i = 0; i < z0.size(); ++i) {
        z[i] = z0[i];
    }

    for (int i = 4; i < n; ++i) {
        y[i] = y[i - 1] + h * z[i - 1];
        z[i] = z[i - 1] + h * (55 * f(x[i - 1], y[i - 1]) - 59 * f(x[i - 2], y[i - 2])
                    + 37 * f(x[i - 3], y[i - 3]) - 9 * f(x[i - 4], y[i - 4])) / 24;
    }

    return y;
}

vector<vector<double>> RRR_inaccuracy(double a, double b, double h, vector<double> x, int n) {  
    vector<double> euler1(n);
    vector<double> runge_kutta1(n);
    vector<double> adams1(n);

    double h2 = h*2;
    int n2 = (b - a) / h2 + 1;
    //cout << n2 << endl;
    vector<double> x2(n);
    for (int i = 0; i < n2; ++i){
        x2[i] = h2 * i;
        //cout << x2[i] << " ";
    }

    vector<double> euler_h = euler(a, b, h, x, n);
    vector<double> euler_2h = euler(a, b, h2, x2, n2);

    vector<double> runge_kutta_h = runge_kutta(a, b, h, x, n).first;
    vector<double> runge_kutta_2h = runge_kutta(a, b, h2, x2, n2).first;
    
    vector<double> adams_h = adams(a, b, h, x, n);
    vector<double> adams_2h = adams(a, b, h2, x2, n2);


    for (int i = 0; i < n2; ++i) {
        // У Эйлера порядок точности р = 1
        euler1[i] = abs(euler_h[i*2] - euler_2h[i]);
        runge_kutta1[i] = abs((runge_kutta_h[i*2] - runge_kutta_2h[i]) / 15);
        adams1[i] = abs((adams_h[i*2] - adams_2h[i]) / 15);
    }

    return {euler1, runge_kutta1, adams1};
}

int main() {
    double a = 0.0;
    double b = 1.0;
    double h = 0.1;

    int n = (b - a) / h + 1;
    int n2 = (b - a) / 2 / h + 1;
    vector<double> x(n), y(n);

    ofstream fout("answer.txt");
    fout.precision(4);
    fout << fixed;

    for (int i = 0; i < n; ++i){
        x[i] = h * i;
        y[i] = accurate_solution(x[i]);
    }

    fout << "Значения x:" << endl;
    for (int i = 0; i < n; ++i){
        fout << x[i] << " ";
    }

    fout << endl << "Точное решение y:" << endl;
    for (int i = 0; i < n; ++i){
        fout << y[i] << " ";
    }
    fout << endl;

    fout << endl << "Метод Эйлера:" << endl;
    for (int i = 0; i < n; ++i){
        fout << euler(a, b, h, x, n)[i] << " ";
    }

    fout << endl << "Метод Рунге-Кутты:" << endl;
    for (int i = 0; i < n; ++i){
        fout << runge_kutta(a, b, h, x, n).first[i] << " ";
    }

    fout << endl << "Метод Адамса:" << endl;
    for (int i = 0; i < n; ++i){
        fout << adams(a, b, h, x, n)[i] << " ";
    }
    fout << endl;

    fout << endl << "Погрешности методом Рунге-Ромберга-Ричардсона" << endl << "Для Эйлера:" << endl;
    for (int i = 0; i < n2; ++i){
        fout << abs(RRR_inaccuracy(a, b, h, x, n)[0][i]) << " ";
    }

    fout << endl << "Для Рунге-Кутты:" << endl;
    for (int i = 0; i < n2; ++i){
        fout << abs(RRR_inaccuracy(a, b, h, x, n)[1][i]) << " ";
    }
    fout << endl << "Для Адамса:" << endl;
    for (int i = 0; i < n2; ++i){
        fout << abs(RRR_inaccuracy(a, b, h, x, n)[2][i]) << " ";
    }
    fout << endl;

    fout << endl << "Погрешности сравнением с точным решением" << endl << "Для Эйлера:" << endl;
    for (int i = 0; i < n; ++i){
        fout << abs(euler(a, b, h, x, n)[i] - y[i]) << " ";
    }

    fout << endl << "Для Рунге-Кутты:" << endl;
    for (int i = 0; i < n; ++i){
        fout << abs(runge_kutta(a, b, h, x, n).first[i] - y[i]) << " ";
    }
    fout << endl << "Для Адамса:" << endl;
    for (int i = 0; i < n; ++i){
        fout << abs(adams(a, b, h, x, n)[i] - y[i]) << " ";
    }

    return 0;
}