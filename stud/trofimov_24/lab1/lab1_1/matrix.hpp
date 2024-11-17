#include <vector>
#include <iostream>
#include <vector>
#include <iostream>
#ifndef LAB1_1_MATRIX_H

typedef    std::vector<std::vector<double> > matrix;

matrix getInverse(matrix& m);
matrix solver(matrix& A, matrix& b);
void print_matrix(const matrix& matrix1);
std::pair<matrix, matrix> LU(matrix& A);
double getDeterminant(matrix m);

#endif //LAB1_1_MATRIX_H
