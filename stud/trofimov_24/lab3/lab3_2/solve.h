//
// Created by владислав трофимов on 11.05.2024.
//

#ifndef LAB3_2_SOLVE_H
#define LAB3_2_SOLVE_H
#include <vector>
#include <iostream>

using namespace std;
using matrix = vector<vector<double> >;
matrix tridiagonal_algorithm(matrix& coefficients, matrix& results);

matrix get_coefs( vector<double>  x, vector<double> y);


#endif //LAB3_2_SOLVE_H
