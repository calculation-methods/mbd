#include <bits/stdc++.h>
using namespace std;

double y(double x) {
    return x / (x*x + 9); 
}

int main() {
    double x_0 = 0, x_k = 2, h1 = 0.5, h2 = 0.25;
    double F1, F2;
    cout << "Rectangle" << endl;
    double x = x_0, res = 0;
    while (x < x_k){
        res += y((2*x + h1)/2);
        x += h1;
    }
    F1 = h1*res;
    x = x_0, res = 0;
    while (x < x_k){
        res += y((2*x + h1)/2);
        x += h2;
    }
    F2 = h2*res;
    cout << "k = 0.5 => result = " << F1 << "\tk = 0.25 => result = " << F2 << "\testimation: " << F1 + (F1 - F2)/(pow((h2/h1), 2) - 1) << endl << endl;
    cout << "Trapeze" << endl;
    x = x_0+h1, res = y(x_0)/2 + y(x_k)/2;
    while (x < x_k){
        res += y(x);
        x += h1;
    }
    F1 = h1 * res;
    x = x_0+h2, res = y(x_0)/2 + y(x_k)/2;
    while (x < x_k){
        res += y(x);
        x += h2;
    }
    F2 = h2 * res;
    cout << "k = 0.5 => result = " << F1 << "\tk = 0.25 => result = " << F2 << "\testimation: " << F1 + (F1 - F2)/(pow((h2/h1), 2) - 1) << endl << endl;
    cout << "Simpson" << endl;    
    x = x_0 + h1, res = y(x_0) + y(x_k);
    bool flag = true;
    while (x < x_k){
        res += y(x) * ((flag) ? 4 : 2);
        x += h1;
        flag = !flag;
    }
    F1 = h1 * res / 3;
    x = x_0 + h2, res = y(x_0) + y(x_k);
    flag = true;
    while (x < x_k){
        res += y(x) * ((flag) ? 4 : 2);
        x += h2;
        flag = !flag;
    }
    F2 = h2 * res / 3;
    cout << "k = 0.5 => result = " << F1 << "\tk = 0.25 => result = " << F2 << "\testimation: " << F1 + (F1 - F2)/(pow((h2/h1), 2) - 1) << endl << endl;
    return 0;
}