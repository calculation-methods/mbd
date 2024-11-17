#include <iostream>
#include <vector>
#ifndef LAB1_3_MATRIX_H

typedef    std::vector<std::vector<double>> matrix;
std::pair<matrix, int> zeidel(matrix& A, matrix& b);
std::pair<matrix,int > simple_iter(matrix& A, matrix& res);
void print_matrix(const matrix& matrix1);

#endif
