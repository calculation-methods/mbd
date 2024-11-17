#include <bits/stdc++.h>
using namespace std;
int main() {
    double t = 1.0;
    vector<double> x = {-1.0, 0.0, 1.0, 2.0, 3.0}, y = {-0.7854, 0.0, 0.78540, 1.1071, 1.249};

    vector<double> first;
    for (int i = 0; i < 4; ++i)
        first.push_back((y[i + 1] - y[i]) / (x[i + 1] - x[i]));

    vector<double> second;
    for (int i = 0; i < 3; ++i) {
        double val = 2 * ((y[i+2]-y[i+1])/(x[i+2]-x[i + 1])-(y[i+1]-y[i])/(x[i+1]-x[i]))/(x[i+2]-x[i]);
        second.push_back(val);
    }

    for (int i = 0; i < 4; ++i) {
        if (x[i] == t) {
            cout << "Правосторонняя первая производная " << first[i - 1] << endl;
            cout << "Правосторонняя первая производная " << first[i] << endl;
            break;
        }
    }

    for (int i = 0; i < 3; ++i) 
        if (x[i] <= t && t <= x[i + 1]) 
            cout << "Вторая производная " << second[i] << endl;

    return 0;
}
