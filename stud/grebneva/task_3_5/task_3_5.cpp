#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

double f(double x){
    return 1 / ((2*x + 7)*(3*x + 4));
}

void methods(double x0, double xk, double& l, double& r, double& m, double& t, double& w, double h){
    int n = (xk - x0) / h + 1;
    vector<double> x(n, 0); 

    for (int i = 0; i < n; ++i){
        x[i] = x0;
        x0 = x0 + h;
    }
    vector<double> y(2*n-1, 0);
    
    for (int i = 0; i < n; i++) {
        y[i] = f(x[i]);
    }

    for (int i = n; i < 2*n - 1; i++){
        y[i] = f((x[i - n] + x[i - n + 1]) / 2);
    }

    for (int i = 0; i < n - 1; i++) {
        l = l + y[i]*h;
        r = r + y[i + 1]*h;
        m = m + y[i + n]*h;
        t = t + (h/2) * (y[i] + y[i + 1]);
        w = w + (h/6)*(y[i] + y[i + 1] + 4*y[i + n]);
    }
}

// Вычисление погрешности методом Рунге-Ромберга-Ричардсона
double RuRoRi(double I_h2, double I_h1){
    double I_wave = 0; // Значение интеграла, уточнённое методом Рунге-Ромберга-Ричардсона на 1 порядок
    double integral = 0.10447;
    I_wave = I_h2 -(I_h1 - I_h2)/3;
    return abs(I_wave - integral); // Абсолютная погрешность
}

int main() {
    double x0, xk, h1, h2;    
    double l = 0, r = 0, m = 0, t = 0, w = 0;   

    ifstream fin("input.txt");
    fin >> x0; 
    fin >> xk;
    fin >> h1;
    fin >> h2;       

    ofstream fout("answer.txt");
    fout.precision(4);
    fout << fixed;

    methods(x0, xk, l, r, m, t, w, h1);
    double ll = l, rr = r, mm = m, tt = t, ww = w;

    l = 0, r = 0, m = 0, t = 0, w = 0;
    methods(x0, xk, l, r, m, t, w, h2);
    

    fout << "Левые прямоугольники с шагом h1: " << ll << endl;
    fout << "Левые прямоугольники с шагом h2: " << l << endl;
    fout << "Погрешность: " << RuRoRi(l, ll) << endl;

    fout << endl << "Правые прямоугольники с шагом h1: " << rr << endl;
    fout << "Правые прямоугольники с шагом h2: " << r << endl;
    fout << "Погрешность: " << RuRoRi(r, rr) << endl;

    fout << endl << "Средние прямоугольники с шагом h1: " << mm << endl;
    fout << "Средние прямоугольники с шагом h2: " << m << endl;
    fout << "Погрешность: " << RuRoRi(m, mm) << endl;

    fout << endl << "Трапеции с шагом h1: " << tt << endl;
    fout << "Трапеции с шагом h2: " << t << endl;
    fout << "Погрешность: " << RuRoRi(t, tt) << endl;

    fout << endl << "Метод Симпсона с шагом h1: " << ww << endl;
    fout << "Метод Симпсона с шагом h2: " << w << endl;
    fout << "Погрешность: " << RuRoRi(w, ww) << endl;
 
    return 0;
}