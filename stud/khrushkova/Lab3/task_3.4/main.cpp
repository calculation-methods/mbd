#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<double> x = {-1.0, -0.5, 0.0, 0.5, 1.0}, y = {-0.36788, -0.30327, 0.0, 0.82436, 2.7183}, p, pp;
    double target = 0.0;

    for (int i = 0; i < 5; i++) 
        p.push_back((y[i + 1] - y[i]) / (x[i + 1] - x[i]));
    for (int i = 0; i < 4; i++)
        pp.push_back(2 * ((y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1]) - (y[i + 1] - y[i]) / (x[i + 1] - x[i])) / (x[i + 2] - x[i]));
    
    for (int i = 0; i < 5; i++)
        if (x[i] == target) {
            cout << "Left derivative = " << p[i - 1] << "\tRight derivative = " << p[i];
            break;
        }

    for (int i = 0; i < 4; i++)
        if (x[i] <= target && target <= x[i + 1]) {
            cout << "\tSecond derivative = " << pp[i] << endl;
            break;
        }
    return 0;
}
