#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

using namespace std;

double func(double x1, double x2){
    return ((asin(x1) + x1) - (asin(x2) + x2)) / (x1 - x2);
}

double func1(double x1, double x2, double x3){
    return (func(x1,x2) - func(x2, x3)) / (x1 - x3);
}

double func2(double x1, double x2, double x3, double x4){
    return (func1(x1, x2, x3) - func1(x2, x3, x4)) / (x1 - x4);
}

vector <double> omega_values(vector <double> x){
    vector <double> omega(4,0);
    omega[0] = (x[0] - x[1]) * (x[0] - x[2]) * (x[0] - x[3]);
    omega[1] = (x[1] - x[0]) * (x[1] - x[2]) * (x[1] - x[3]);
    omega[2] = (x[2] - x[0]) * (x[2] - x[1]) * (x[2] - x[3]);
    omega[3] = (x[3] - x[0]) * (x[3] - x[1]) * (x[3] - x[2]);
    return omega;
}

void Lagrange_method(vector <double> x, double X, vector <double> &L, vector <double> &y, vector <double> &delta){
    vector <vector <double>> table(5, vector<double>(4,0));
    vector <double> omega = omega_values(x); 
    for (int i = 0; i < x.size(); i++){
        table[0][i] = x[i];
        table[1][i] = asin(x[i]) + x[i];
        table[2][i] = omega[i];
        table[3][i] = (asin(x[i]) + x[i]) / omega[i];
        table[4][i] = X - x[i];
    }

    for (int i = 0; i < x.size(); i++){
        L[i] = (table[3][0] * (x[i] - table[0][1]) * (x[i] - table[0][2]) * (x[i] - table[0][3]) \
         + table[3][1] * (x[i] - table[0][0]) * (x[i] - table[0][2]) * (x[i] - table[0][3]) \
         + table[3][2] * (x[i] - table[0][0]) * (x[i] - table[0][1]) * (x[i] - table[0][3]) \
         + table[3][3] * (x[i] - table[0][0]) * (x[i] - table[0][1]) * (x[i] - table[0][2]));
    }

    for (int i = 0; i < x.size(); i++){
        y[i] = asin(x[i]) + x[i];
    }

    for (int i = 0; i < x.size(); i++){
        delta[i] = fabs(y[i] - L[i]);
    }
}


void Newton_method(vector <double> x, double X, vector <double> &P, vector <double> &y, vector <double> &delta){
    vector <vector <double>> table(5, vector<double>(4,0));
    for (int i = 0; i < x.size(); i++){
        table[0][i] = x[i];
        table[1][i] = asin(x[i]) + x[i];
        if (i < 3){
            table[2][i] = func(x[i], x[i+1]);
        }
        if (i < 2){
            table[3][i] = func1(x[i], x[i+1], x[i+2]);
        }
    }
    table[4][0] = func2(x[0], x[1], x[2], x[3]);


    for (int i = 0; i < x.size(); i++){
        P[i] = (table[1][0] + table[2][0] * (x[i] - table[0][0]) + table[3][0] * (x[i] - table[0][0]) * (x[i] - table[0][1]) \
         + table[4][0] * (x[i] - table[0][0]) * (x[i] - table[0][1]) * (x[i] - table[0][2]));
    }

    for (int i = 0; i < x.size(); i++){
        y[i] = asin(x[i]) + x[i];
    }

    for (int i = 0; i < x.size(); i++){
        delta[i] = fabs(y[i] - P[i]);
    }
}

int main(){
    ofstream fout("answer1.txt");
    fout.precision(2);
    fout << fixed;
    int n = 4;
    vector <double> x1 = {-0.4, -0.1, 0.2, 0.5};
    vector <double> x2 = {-0.4, 0, 0.2, 0.5};
    double root = 0.1;
    vector <double> L(n,0), y(n,0), delta(n,0), P(n,0);
    fout << "Lagrange method" << endl;
    Lagrange_method(x1, root, L, y, delta);
    fout << "for x1" << endl << "L(x)\n" << "[";
    for (int i = 0; i < L.size(); i++) fout << L[i] << "\t";
    fout << "]\n" << "y(x)\n" << "["; 
    for (int i = 0; i < y.size(); i++) fout << y[i] << "\t";
    fout << "]\n" << "delta(x)\n" << "["; 
    for (int i = 0; i < delta.size(); i++) fout << delta[i] << "\t";
    fout << "]\n"; 
    Lagrange_method(x2, root, P, y, delta);
    fout << "\n\nfor x2" << endl << "L(x)\n" << "[";
    for (int i = 0; i < L.size(); i++) fout << L[i] << "\t";
    fout << "]\n" << "y(x)\n" << "["; 
    for (int i = 0; i < y.size(); i++) fout << y[i] << "\t";
    fout << "]\n" << "delta(x)\n" << "["; 
    for (int i = 0; i < delta.size(); i++) fout << delta[i] << "\t";
    fout << "]\n"; 
    fout << "\n\nNewton method" << endl;
    Newton_method(x1, root, P, y, delta);
    fout << "for x1" << endl << "P(x)\n" << "[";
    for (int i = 0; i < P.size(); i++) fout << P[i] << "\t";
    fout << "]\n" << "y(x)\n" << "["; 
    for (int i = 0; i < y.size(); i++) fout << y[i] << "\t";
    fout << "]\n" << "delta(x)\n" << "["; 
    for (int i = 0; i < delta.size(); i++) fout << delta[i] << "\t";
    fout << "]\n"; 
    Newton_method(x2, root, P, y, delta);
    fout << "\n\nfor x2" << endl << "P(x)\n" << "[";
    for (int i = 0; i < P.size(); i++) fout << P[i] << "\t";
    fout << "]\n" << "y(x)\n" << "["; 
    for (int i = 0; i < y.size(); i++) fout << y[i] << "\t";
    fout << "]\n" << "delta(x)\n" << "["; 
    for (int i = 0; i < delta.size(); i++) fout << delta[i] << "\t";
    fout << "]\n"; 
}