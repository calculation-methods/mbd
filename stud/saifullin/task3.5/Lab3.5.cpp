#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

using namespace std;

double func(double x){
    return (x * x) / (625 - x * x * x * x); 
}


double rectangle_method(double x0, double xk, double h){
    double F = 0;
    double n = (int) ((xk - x0) / h);
    n += 1;
    vector <double> x_values(n, 0);
    for (int i = 0; i < n; i++){
        x_values[i] = x0 + h*i;
    }
    for (int i = 1; i < n; i++){
        F += h * func((x_values[i] + x_values[i-1])/2);
    }
    return F;
}

double trapez_method(double x0, double xk, double h){
    double F = 0;
    double n = (int) ((xk - x0) / h);
    n += 1;
    vector <double> x_values(n, 0);
    for (int i = 0; i < n; i++){
        x_values[i] = x0 + h*i;
    }
    for (int i = 1; i < n; i++){
        F += (func(x_values[i]) + func(x_values[i-1])) / 2 * h;
    }
    return F;
}

double simps_method(double x0, double xk, double h){
    int n = (int)((xk - x0) / h);
    double F = 0;
    for (int i = 0; i < n; i++) {
        double x1 = x0 + i * h;
        double x2 = x0 + (i + 1) * h;
        double x3 = x0 + (i + 0.5) * h;
        F += (h / 6) * (func(x1) + 4 * func(x3) + func(x2));
    }
    return F;
}

vector <double> runge_romb_rich(double x0, double xk, double h){
    double F = 0;
    vector <double> results(3,0);
    results[0] = rectangle_method(x0, xk, h) + (rectangle_method(x0, xk, h/2) - rectangle_method(x0, xk, h/2))/(1-0.5*0.5);
    results[1] = trapez_method(x0, xk, h) + (trapez_method(x0, xk, h/2) - trapez_method(x0, xk, h/2))/(1-0.5*0.5);
    results[2] = simps_method(x0, xk, h) + (simps_method(x0, xk, h/2) - simps_method(x0, xk, h/2))/(1-0.5*0.5*0.5*0.5);
    return results;
}


int main(){
    ofstream fout("answer5.txt");
    // fout.precision(2);
    // fout << fixed;
    int n = 5;
    double x0 = 0, xk = 4, h1 = 1.0, h2 = 0.5;
    double integral = 0.042387134;
    fout << "Rectangle with h = 1\n";
    fout << rectangle_method(x0, xk, h1);
    fout << "\nRectangle with h = 0.5\n";
    fout << rectangle_method(x0, xk, h2);
    fout << "\n\nTrapez with h = 1\n";
    fout << trapez_method(x0,xk,h1);
    fout << "\nTrapez with h = 0.5\n";
    fout << trapez_method(x0,xk,h2);
    fout << "\n\nSimpson with h = 1\n";
    fout << simps_method(x0,xk,h1);
    fout << "\nSimpson with h = 0.5\n";
    fout << simps_method(x0,xk,h2) << endl << endl;
    vector <double> RRR = runge_romb_rich(x0, xk, h1);
    vector <double> RRR2 = runge_romb_rich(x0,xk,h2);
    fout << "with Runge-Romberg-Richardson method" << endl;
    fout << "Rectangle with h = 1\n";
    fout << RRR[0];
    fout << "\nпогрешность:" << fabs(integral - RRR[0]);
    fout << "\nRectangle with h = 0.5\n";
    fout << RRR2[0];
    fout << "\nпогрешность:" << fabs(integral - RRR2[0]);
    fout << "\n\nTrapez with h = 1\n";
    fout << RRR[1];
    fout << "\nпогрешность:" << fabs(integral - RRR[1]);
    fout << "\nTrapez with h = 0.5\n";
    fout << RRR2[1];
    fout << "\nпогрешность:" << fabs(integral - RRR2[1]);
    fout << "\n\nSimpson with h = 1\n";
    fout << RRR[2];
    fout << "\nпогрешность:" << fabs(integral - RRR[2]);
    fout << "\nSimpson with h = 0.5\n";
    fout << RRR2[2];
    fout << "\nпогрешность:" << fabs(integral - RRR2[2]);
}