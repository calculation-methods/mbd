#include <vector>
#include <iostream>

#include "solve.h"



int main() {
    double x_marked = 0.8;
    vector<double> x = {0.1, 0.5, 0.9, 1.3, 1.7};
    vector<double> y = {100.00,	4.0,	1.2346,	0.59172	,0.34602};
    int n = x.size() - 1;

    matrix coef = get_coefs(x,y);
    vector<double> coeff_a = coef[0];
    vector<double> coeff_b = coef[1];
    vector<double> coeff_c = coef[2];
    vector<double> coeff_d = coef[3];

    for (int i = 0; i < n; ++i) {
        if (x[i] <= x_marked && x_marked <= x[i + 1]) {
            double res = coeff_a[i] + coeff_b[i]*(x_marked-x[i]) + coeff_c[i]*(x_marked-x[i])*(x_marked-x[i]) + coeff_d[i]*(x_marked-x[i])*(x_marked-x[i])*(x_marked-x[i]);
            cout << "Result = " << res << endl;
            break;
        }
    }



    return 0;
}