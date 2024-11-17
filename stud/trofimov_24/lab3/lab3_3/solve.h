//
// Created by владислав трофимов on 11.05.2024.
//

#ifndef LAB3_3_SOLVE_H
#define LAB3_3_SOLVE_H
#include <vector>
#include <iostream>

using namespace std;
using matrix = vector<vector<double> >;

matrix solve(matrix& coefficients, matrix& results);
vector<double> get_sums(vector<double> x,vector<double> y);


#endif //LAB3_3_SOLVE_H
