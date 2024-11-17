#include <vector>
#include <iostream>
#include "solve.h"

using namespace std;


int main() {
    vector<double> x = {0.1	,0.5	,0.9	,1.3	,1.7,	2.1};
    vector<double> y = {100.0	,4.0	,1.2346,	0.59172	,0.34602,	0.22676};

    double n = x.size();

    vector<double> sums = get_sums(x,y);
    double sum_x = sums[0];
    double sum_x2 = sums[1];
    double sum_x3 = sums[2];
    double sum_x4 = sums[3];
    double sum_y = sums[4];
    double sum_yx = sums[5];
    double sum_yx2 = sums[6];

    matrix matr = {
            {n, sum_x},
            {sum_x, sum_x2}
    };

    matrix root = {
            {sum_y},
            {sum_yx}
    };

    matrix decisions = solve(matr, root);

    cout << "F1(x) = (" << decisions[0][0] << ") + (" << decisions[1][0] << ")*x" << endl;
    double loss_f1 = 0;
    for (int i = 0; i < n; i++)
        loss_f1 += pow(decisions[0][0] + decisions[1][0] * x[i] - y[i], 2);
    cout << "Loss = " << loss_f1 << endl << endl;


    matr = {
            {n, sum_x, sum_x2},
            {sum_x, sum_x2, sum_x3},
            {sum_x2, sum_x3, sum_x4}
    };

    root = {
            {sum_y},
            {sum_yx},
            {sum_yx2}
    };

    decisions = solve(matr, root);

    cout << "F2(x) = (" << decisions[0][0] << ") + (" << decisions[1][0] << ")*x + (" << decisions[2][0] << ")*x^2" << endl;
    double loss_f2 = 0;
    for (int i = 0; i < n; i++)
        loss_f2 += pow(decisions[0][0] + decisions[1][0] * x[i] + decisions[2][0] * pow(x[i], 2) - y[i], 2);
    cout << "Loss = " << loss_f2 << endl;

    return 0;
}