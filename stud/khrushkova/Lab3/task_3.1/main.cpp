#include <bits/stdc++.h>
using namespace std;

const double PI = 3.1415926535;

double f(double x){
    return x*sin(x);
}


int main() {
    vector<double> x_vector = {0.0, PI/6, 5*PI/12, PI/2};
    double target = PI / 4;

    vector<pair<double, double>> crds;
    for (double x : x_vector)
        crds.emplace_back(x, f(x));

    vector<double> coefs(crds.size());
    int n = crds.size();


    for (int i = 0; i < n; ++i)
        coefs[i] = crds[i].second;
    for (int i = 1; i < n; ++i)
        for (int j = n - 1; j > i-1; --j) 
            coefs[j] = (coefs[j] - coefs[j - 1]) / (crds[j].first - crds[j - i].first);
    for (int i = 1; i < n; ++i) 
        for (int j = 0; j < i; ++j) 
            coefs[i] *= target - crds[j].first;
    
    double lagrange = 0;
    for(auto val: coefs)
        lagrange += val;


    for (int i = 0; i < n; ++i)
        coefs[i] = crds[i].second;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j)
                coefs[i] /= crds[i].first - crds[j].first;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j)
                coefs[i] *= target - crds[j].first;
    double newton = 0;
    for(auto val: coefs)
        newton += val;


    cout << "Lagrange algo" << "\nResult: " << lagrange << "\nThe absolute difference: " << abs(lagrange - f(target)) << endl << endl;
    cout << "Newton algo" << "\nResult: " << newton << "\nThe absolute difference: " << abs(newton - f(target)) << endl;
    return 0;
}