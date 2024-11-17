#include <bits/stdc++.h>

using namespace std;

double f(double x) {
    return acos(x);
}

double lagrange_interpolation(double x, const vector<pair<double, double>>& coords) {
    vector<double> coefficients(coords.size());
    int n = coords.size();

    for (int i = 0; i < n; ++i)
        coefficients[i] = coords[i].second;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j)
                coefficients[i] /= coords[i].first - coords[j].first;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j)
                coefficients[i] *= x - coords[j].first;

    return accumulate(coefficients.begin(), coefficients.end(), 0.0);
}

double newton_interpolation(double x, const vector<pair<double, double>>& coords) {
    vector<double> coefficients(coords.size());
    int n = coords.size();

    for (int i = 0; i < n; ++i)
        coefficients[i] = coords[i].second;

    for (int i = 1; i < n; ++i)
        for (int j = n - 1; j > i-1; --j) 
            coefficients[j] = (coefficients[j] - coefficients[j - 1]) / (coords[j].first - coords[j - i].first);

    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            coefficients[i] *= x - coords[j].first;
        }
    }

    return accumulate(coefficients.begin(), coefficients.end(), 0.0);
}

int main() {
    vector<double> x_vect = {-0.4, -0.1, 0.2, 0.5};
    double x_marked = 0.1;
    vector<pair<double, double>> cord;

    for (double x : x_vect)
        cord.emplace_back(x, f(x));

    cout << "\nThe Lagrange polynomial\n";
    cout << "\tResult: " << lagrange_interpolation(x_marked, cord) << endl;
    cout << "\tLoss: " << abs(lagrange_interpolation(x_marked, cord) - f(x_marked)) << endl;
    cout << "\nThe Newton polynomial\n";
    cout << "\tResult: " << newton_interpolation(x_marked, cord) << endl;
    cout << "\tLoss: " << abs(newton_interpolation(x_marked, cord) - f(x_marked)) << endl;

    return 0;
}
