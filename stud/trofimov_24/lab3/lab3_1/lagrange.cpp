#include <vector>
#include <iostream>
#include <numeric>
#include "lagrange.h"

using namespace std;


double Lagrange::result(double x, const vector<pair<double, double>>& coords) {
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