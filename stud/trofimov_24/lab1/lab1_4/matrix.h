
#include <iostream>
#include <vector>
#include <algorithm>
#ifndef LAB1_1_4MATRIX_H

typedef    std::vector<std::vector<double>> matrix;
std::pair<matrix,matrix> solve(matrix A,double eps0);
void print_matrix(const matrix& matrix1);

#endif
