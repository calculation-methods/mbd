#include <bits/stdc++.h>
using namespace std;

double newton(double x, const vector<pair<double, double>>& xy) {
    vector<double> coefs(xy.size());
    int n = xy.size();
    for (int i = 0; i < n; ++i)
        coefs[i] = xy[i].second;
    for (int i = 1; i < n; ++i)
        for (int j = n - 1; j > i-1; --j) 
            coefs[j] = (coefs[j] - coefs[j - 1]) / (xy[j].first - xy[j - i].first);
    for (int i = 1; i < n; ++i) 
        for (int j = 0; j < i; ++j) 
            coefs[i] *= x - xy[j].first;
    double res = 0;
    for(auto val: coefs)
        res += val;
    return res;
}

double lagrange(double x, const vector<pair<double, double>>& xy) {
    vector<double> coefs(xy.size());
    int n = xy.size();
    for (int i = 0; i < n; ++i)
        coefs[i] = xy[i].second;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j)
                coefs[i] /= xy[i].first - xy[j].first;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j)
                coefs[i] *= x - xy[j].first;
    double res = 0;
    for(auto val: coefs)
        res += val;
    return res;
}

int main() {
    vector<double> input_x = {0, 1.7, 4.0, 5.1};
    double star_x = 3.0;
    vector<pair<double, double>> xy;
    for (double x : input_x)
        xy.emplace_back(x, sqrt(x) + x);
    cout << "Lagrange" << "\t\tResult: " << lagrange(star_x, xy) << "\t\tLoss: " << abs(lagrange(star_x, xy) - sqrt(star_x) - star_x) << endl;
    cout << "Newton  " << "\t\tResult: " << newton(star_x, xy) << "\t\tLoss: " << abs(newton(star_x, xy) - sqrt(star_x) - star_x) << endl;
    return 0;
}