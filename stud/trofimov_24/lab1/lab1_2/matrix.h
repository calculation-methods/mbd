
#include <vector>

#ifndef LAB1_1_MATRIX_H

typedef    std::vector<std::vector<double>> matrix;
matrix solver(matrix& A, matrix& b);
void print_matrix(const matrix& matrix1);

#endif //LAB1_1_MATRIX_H