//
// Created by владислав трофимов on 11.05.2024.
//

#include "newton.h"

#include <vector>
#include <iostream>
#include <numeric>
#include <math.h>





double Newton::result(double x, const vector<pair<double, double>>& coords) {
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